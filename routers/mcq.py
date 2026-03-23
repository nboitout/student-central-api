from fastapi import APIRouter, HTTPException
from models.mcq import MCQGenerateRequest, MCQEvaluateRequest
from services import openai_service, cosmos_service
from services.blob_service import get_blob_sas_url
from services.pdf_service import render_pdf_pages_as_images

router = APIRouter(prefix="/api/mcq", tags=["mcq"])


@router.post("/generate")
async def generate_mcq(payload: MCQGenerateRequest):
    """
    Generate one MCQ question grounded strictly in the course PDF.
    Renders PDF pages as images so GPT sees charts, diagrams, and visuals.
    Requires the course to have a PDF attached — returns 400 if not.
    """

    # Step 1 — Get PDF URL from payload or Cosmos DB
    course = None
    pdf_url = payload.pdfUrl

    if not pdf_url and payload.courseId:
        course = await cosmos_service.get_course(payload.courseId)
        if course:
            pdf_url = course.get("pdfUrl")

    # Step 2 — Block if no PDF attached
    if not pdf_url:
        raise HTTPException(
            status_code=400,
            detail="No PDF attached to this course. Please upload a course document before generating questions."
        )

    # Step 3 — Get course title
    course_title = payload.courseTitle
    if not course_title:
        if course:
            course_title = course.get("title", "")
        elif payload.courseId:
            course = await cosmos_service.get_course(payload.courseId)
            if course:
                course_title = course.get("title", "")

    if not course_title:
        raise HTTPException(status_code=400, detail="Course title is required.")

    # Step 4 — Generate SAS URL for secure PDF access
    try:
        sas_url = get_blob_sas_url(pdf_url, expiry_hours=1)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to access course PDF: {str(e)}"
        )

    # Step 5 — Render PDF pages as images (max 10 pages, 150 DPI)
    try:
        pdf_images = await render_pdf_pages_as_images(
            sas_url=sas_url,
            max_pages=10,
            dpi=150,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render PDF pages: {str(e)}"
        )

    if not pdf_images:
        raise HTTPException(
            status_code=422,
            detail="Could not render any pages from the PDF."
        )

    # Step 6 — Generate MCQ grounded in PDF page images
    try:
        mcq = await openai_service.generate_mcq(
            course_title=course_title,
            pdf_images=pdf_images,
            course_id=payload.courseId,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate MCQ: {str(e)}"
        )

    return mcq


@router.post("/evaluate")
async def evaluate_reasoning(payload: MCQEvaluateRequest):
    """
    Evaluate the quality of a student's reasoning after an MCQ answer.
    Returns a reasoning signal: Strong / Fragile / Partial misconception / Low mastery.
    """
    if not payload.studentExplanation or len(payload.studentExplanation.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="A written explanation of at least 10 characters is required for evaluation."
        )

    try:
        signal = await openai_service.evaluate_reasoning(
            question=payload.question,
            options=payload.options,
            correct_index=payload.correctIndex,
            selected_index=payload.selectedIndex,
            student_explanation=payload.studentExplanation,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate reasoning: {str(e)}"
        )

    return signal
