import random
from fastapi import APIRouter, HTTPException
from models.mcq import MCQGenerateRequest, MCQEvaluateRequest, MCQQuestion, MCQOption
from services import openai_service, cosmos_service

router = APIRouter(prefix="/api/mcq", tags=["mcq"])


@router.get("/bank/{course_id}")
async def get_mcq_bank(course_id: str, userId: str = "nicolas"):
    """
    Return the full MCQ bank for a course.
    Faculty can use this to review all questions.
    """
    mcqs = await cosmos_service.get_mcq_bank(course_id)
    return {"mcqs": mcqs, "count": len(mcqs), "courseId": course_id}


@router.post("/generate")
async def get_next_mcq(payload: MCQGenerateRequest):
    """
    Return one MCQ from the pre-generated bank for this course.
    Questions are served in random order.
    Returns 400 if no PDF attached, 202 if still generating, 404 if bank is empty.
    """

    # Fetch course to check status
    course = await cosmos_service.get_course(payload.courseId)

    if not course:
        raise HTTPException(status_code=404, detail="Course not found.")

    if not course.get("pdfUrl"):
        raise HTTPException(
            status_code=400,
            detail="No PDF attached to this course. Please upload a course document first."
        )

    mcq_status = course.get("mcqStatus", "none")

    if mcq_status == "none":
        raise HTTPException(
            status_code=400,
            detail="MCQ generation has not been triggered for this course."
        )

    if mcq_status == "generating":
        raise HTTPException(
            status_code=202,
            detail="Questions are being generated from your document. Please try again in a moment."
        )

    if mcq_status == "failed":
        raise HTTPException(
            status_code=500,
            detail="MCQ generation failed for this course. Please re-upload the document."
        )

    # Fetch the MCQ bank
    bank = await cosmos_service.get_mcq_bank(payload.courseId)

    if not bank:
        raise HTTPException(
            status_code=404,
            detail="No questions found for this course. The bank may still be loading."
        )

    # Pick a random question
    item = random.choice(bank)

    options = [
        MCQOption(letter=opt["letter"], text=opt["text"])
        for opt in item["options"]
    ]

    return MCQQuestion(
        question=item["question"],
        options=options,
        correctIndex=item["correctIndex"],
        explanation=item["explanation"],
        courseId=payload.courseId,
    )


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
