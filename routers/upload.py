from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from services.blob_service import upload_pdf, get_blob_sas_url
from services.pdf_service import render_pdf_pages_as_images
from services.openai_service import generate_mcq_bank
from services import cosmos_service
from models.mcq import StoredMCQ
from models.course import CourseUpdate

router = APIRouter(prefix="/api/upload", tags=["upload"])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MCQ_COUNT = 10


async def _generate_and_store_mcqs(
    course_id: str,
    user_id: str,
    pdf_url: str,
    course_title: str,
):
    """
    Background task: render PDF pages as images, generate 10 MCQs,
    store them in Cosmos DB, and update the course mcqStatus.
    """
    try:
        # Mark as generating
        await cosmos_service.update_course(
            course_id,
            CourseUpdate(mcqStatus="generating"),
            user_id=user_id,
        )

        # Generate SAS URL for PDF access
        sas_url = get_blob_sas_url(pdf_url, expiry_hours=2)

        # Render PDF pages as images
        pdf_images = await render_pdf_pages_as_images(
            sas_url=sas_url,
            max_pages=10,
            dpi=150,
        )

        if not pdf_images:
            raise ValueError("No pages rendered from PDF")

        # Generate MCQ bank
        questions = await generate_mcq_bank(
            course_title=course_title,
            pdf_images=pdf_images,
            course_id=course_id,
            count=MCQ_COUNT,
        )

        # Store MCQs in Cosmos DB
        stored_mcqs = [
            StoredMCQ(
                courseId=course_id,
                userId=user_id,
                question=q.question,
                options=q.options,
                correctIndex=q.correctIndex,
                explanation=q.explanation,
            )
            for q in questions
        ]
        saved_count = await cosmos_service.save_mcq_bank(stored_mcqs)

        # Update course: mcqStatus=ready, mcqCount=saved
        await cosmos_service.update_course(
            course_id,
            CourseUpdate(mcqStatus="ready", mcqCount=saved_count),
            user_id=user_id,
        )

        print(f"MCQ generation complete for course {course_id}: {saved_count} questions stored")

    except Exception as e:
        print(f"MCQ generation failed for course {course_id}: {e}")
        try:
            await cosmos_service.update_course(
                course_id,
                CourseUpdate(mcqStatus="failed"),
                user_id=user_id,
            )
        except Exception:
            pass


@router.post("")
async def upload_course_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file to Azure Blob Storage.
    Returns the permanent blob URL to store with the course record.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    blob_url = await upload_pdf(
        file_bytes=contents,
        original_filename=file.filename,
    )

    return {
        "url": blob_url,
        "filename": file.filename,
        "size": len(contents),
    }


@router.post("/trigger-mcq-generation")
async def trigger_mcq_generation(
    background_tasks: BackgroundTasks,
    course_id: str,
    user_id: str = "nicolas",
    pdf_url: str | None = None,
    course_title: str | None = None,
):
    """
    Trigger background MCQ generation for a course after PDF is attached.
    Called by the frontend immediately after PATCH /api/courses/:id/pdf succeeds.
    Returns immediately — generation happens in the background.
    pdf_url and course_title are optional — fetched from Cosmos DB if not provided.
    """
    # Fetch course to fill in missing params and validate PDF exists
    course = await cosmos_service.get_course(course_id, user_id=user_id)
    if not course:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Course not found.")

    resolved_pdf_url = pdf_url or course.get("pdfUrl")
    resolved_title = course_title or course.get("title", "")

    if not resolved_pdf_url:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No PDF attached to this course.")

    # Check if MCQs already exist and are ready
    existing = await cosmos_service.get_mcq_bank(course_id)
    if existing and course.get("mcqStatus") == "ready":
        return {"status": "already_generated", "count": len(existing)}

    # Queue background task
    background_tasks.add_task(
        _generate_and_store_mcqs,
        course_id=course_id,
        user_id=user_id,
        pdf_url=resolved_pdf_url,
        course_title=resolved_title,
    )

    return {"status": "generating", "message": "MCQ generation started in background"}
