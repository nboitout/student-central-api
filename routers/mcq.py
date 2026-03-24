import random
from fastapi import APIRouter, HTTPException
from models.mcq import MCQGenerateRequest, MCQEvaluateRequest, MCQQuestion, MCQOption
from services import openai_service, cosmos_service
from services.blob_service import get_blob_sas_url

router = APIRouter(prefix="/api/mcq", tags=["mcq"])


@router.get("/bank/{course_id}")
async def get_mcq_bank(course_id: str, userId: str = "nicolas"):
    """Return the full MCQ bank for a course. Faculty use."""
    mcqs = await cosmos_service.get_mcq_bank(course_id)
    return {"mcqs": mcqs, "count": len(mcqs), "courseId": course_id}


@router.get("/bank/{course_id}/{mcq_id}/slide")
async def get_slide_sas_url(course_id: str, mcq_id: str):
    """
    Return a short-lived SAS URL for the slide image associated with an MCQ.
    Frontend uses this to render the related PDF page alongside the question.
    SAS URL is valid for 2 hours.
    """
    # Fetch the MCQ from the bank
    bank = await cosmos_service.get_mcq_bank(course_id)
    mcq = next((m for m in bank if m["id"] == mcq_id), None)

    if not mcq:
        raise HTTPException(status_code=404, detail="MCQ not found.")

    slide_url = mcq.get("slideImageUrl")
    if not slide_url:
        raise HTTPException(status_code=404, detail="No slide image available for this question.")

    try:
        sas_url = get_blob_sas_url(slide_url, expiry_hours=2)
        return {
            "sasUrl": sas_url,
            "pageNumber": mcq.get("pageNumber", 0),
            "mcqId": mcq_id,
            "courseId": course_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate slide URL: {str(e)}")


@router.post("/generate")
async def get_next_mcq(payload: MCQGenerateRequest):
    """
    Return one MCQ from the pre-generated bank.
    Response includes mcqId, pageNumber, and slideImageUrl (private — use /slide for SAS).
    """
    course = await cosmos_service.get_course(payload.courseId)

    if not course:
        raise HTTPException(status_code=404, detail="Course not found.")

    if not course.get("pdfUrl"):
        raise HTTPException(status_code=400, detail="No PDF attached to this course.")

    mcq_status = course.get("mcqStatus", "none")

    if mcq_status in ("none", "generating"):
        raise HTTPException(
            status_code=202,
            detail="Questions are being generated from your document. Please try again in a moment."
        )

    if mcq_status == "failed":
        raise HTTPException(status_code=500, detail="MCQ generation failed. Please re-upload the document.")

    bank = await cosmos_service.get_mcq_bank(payload.courseId)

    if not bank:
        raise HTTPException(status_code=404, detail="No questions found for this course.")

    item = random.choice(bank)

    options = [MCQOption(letter=opt["letter"], text=opt["text"]) for opt in item["options"]]

    return MCQQuestion(
        mcqId=item["id"],
        question=item["question"],
        options=options,
        correctIndex=item["correctIndex"],
        explanation=item["explanation"],
        courseId=payload.courseId,
        pageNumber=item.get("pageNumber"),
        slideImageUrl=item.get("slideImageUrl"),  # private URL — frontend must call /slide for SAS
    )


@router.post("/evaluate")
async def evaluate_reasoning(payload: MCQEvaluateRequest):
    """Evaluate student reasoning. Returns Strong / Fragile / Partial misconception / Low mastery."""
    if not payload.studentExplanation or len(payload.studentExplanation.strip()) < 10:
        raise HTTPException(status_code=400, detail="Explanation of at least 10 characters required.")

    try:
        signal = await openai_service.evaluate_reasoning(
            question=payload.question,
            options=payload.options,
            correct_index=payload.correctIndex,
            selected_index=payload.selectedIndex,
            student_explanation=payload.studentExplanation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate reasoning: {str(e)}")

    return signal
