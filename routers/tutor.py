from fastapi import APIRouter, HTTPException
from models.tutor import TutorProbeRequest, TutorReplyRequest, TutorResponse
from services import tutor_service

router = APIRouter(prefix="/api/tutor", tags=["tutor"])


@router.post("/probe", response_model=TutorResponse)
async def probe(payload: TutorProbeRequest):
    """
    Opening Socratic probe — called once when the student clicks 'Discuss with AI'.
    Returns the tutor's first question to surface the student's reasoning.
    Never reveals the correct answer. Never evaluates.
    """
    try:
        message = await tutor_service.generate_probe(
            question=payload.question,
            options=payload.options,
            correct_index=payload.correctIndex,
            selected_index=payload.selectedIndex,
            is_correct=payload.isCorrect,
            explanation=payload.explanation,
            language=payload.language,
        )
        return TutorResponse(message=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tutor probe failed: {str(e)}")


@router.post("/reply", response_model=TutorResponse)
async def reply(payload: TutorReplyRequest):
    """
    Follow-up Socratic reply — called each time the student sends a message.
    Receives full conversation history and returns the tutor's next question.
    Progresses the Socratic arc across up to 5 turns.
    Never reveals the correct answer. Never evaluates.
    """
    if not payload.history:
        raise HTTPException(status_code=400, detail="History must not be empty for a reply.")

    try:
        message = await tutor_service.generate_reply(
            question=payload.question,
            options=payload.options,
            correct_index=payload.correctIndex,
            selected_index=payload.selectedIndex,
            is_correct=payload.isCorrect,
            explanation=payload.explanation,
            language=payload.language,
            history=[m.model_dump() for m in payload.history],
        )
        return TutorResponse(message=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tutor reply failed: {str(e)}")
