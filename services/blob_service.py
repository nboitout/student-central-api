from fastapi import APIRouter, HTTPException
from models.mcq import MCQGenerateRequest, MCQEvaluateRequest
from services import openai_service, cosmos_service
from services.blob_service import get_blob_sas_url

router = APIRouter(prefix="/api/mcq", tags=["mcq"])


@router.post("/generate")
async def generate_mcq(payload: MCQGenerateRequest):
    """
    Generate one MCQ question for a given course.
    If the course has a PDF attached, uses it as context for GPT-4o.
    Falls back to course title only if no PDF is available.
    """
    # Fetch course to get PDF URL if not provided
    pdf_sas_url = None

    if payload.pdfUrl:
        # Generate a short-lived SAS URL for secure GPT-4o access
        try:
            pdf_sas_url = get_blob_sas_url(payload.pdfUrl)
        except Exception:
            # If SAS generation fails, proceed without PDF
            pdf_sas_url = None
    elif payload.courseId:
        course = await cosmos_service.get_course(payload.courseId)
        if course and course.get("pdfUrl"):
            try:
                pdf_sas_url = get_blob_sas_url(course["pdfUrl"])
            except Exception:
                pdf_sas_url = None

    # Determine course title
    course_title = payload.courseTitle
    if not course_title and payload.courseId:
        course = await cosmos_service.get_course(payload.courseId)
        if course:
            course_title = course.get("title", "")

    if not course_title:
        raise HTTPException(status_code=400, detail="Course title or courseId is required.")

    try:
        mcq = await openai_service.generate_mcq(
            course_title=course_title,
            pdf_sas_url=pdf_sas_url,
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
