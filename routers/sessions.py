from fastapi import APIRouter, HTTPException
from models.session import (
    SessionCreateRequest,
    SessionAnswerRequest,
    SessionExplanationRequest,
    SessionChatRequest,
)
from services import session_service
from services.openai_service import evaluate_reasoning

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("")
async def create_session(payload: SessionCreateRequest):
    """
    Create a new session and pre-select 5 questions from the bank.
    Excludes questions already seen by this user in previous completed sessions.
    Returns the session ID and the first question to display.
    """
    try:
        session, drawn = await session_service.create_session(
            course_id=payload.courseId,
            user_id=payload.userId,
            mode=payload.mode,
            language=payload.language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    first = session.questions[0]
    return {
        "sessionId": session.id,
        "mode": session.mode,
        "language": session.language,
        "totalQuestions": len(session.questions),
        "question": {
            "position": first.position,
            "mcqId": first.mcqId,
            "question": first.question,
            "options": first.options,
            "correctIndex": first.correctIndex,
            "pageNumber": first.pageNumber,
        }
    }


@router.get("/{session_id}")
async def get_session(session_id: str, courseId: str):
    """Get the full session document."""
    session = await session_service.get_session(session_id, courseId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@router.get("")
async def list_sessions(courseId: str, userId: str = "nicolas"):
    """List all sessions for a user/course. Used by the faculty dashboard."""
    return await session_service.list_sessions(courseId, userId)


@router.get("/{session_id}/question/{position}")
async def get_question(session_id: str, position: int, courseId: str):
    """
    Return the question at a given position.
    Used when the frontend advances to the next question.
    """
    session = await session_service.get_session(session_id, courseId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    question = next(
        (q for q in session["questions"] if q["position"] == position), None
    )
    if not question:
        raise HTTPException(status_code=404, detail=f"Question {position} not found.")

    return {
        "position": question["position"],
        "mcqId": question["mcqId"],
        "question": question["question"],
        "options": question["options"],
        "correctIndex": question["correctIndex"],
        "pageNumber": question.get("pageNumber"),
    }


@router.patch("/{session_id}/answer")
async def record_answer(session_id: str, courseId: str, payload: SessionAnswerRequest):
    """Record the student's answer for a question."""
    updated = await session_service.record_answer(
        session_id=session_id,
        course_id=courseId,
        position=payload.position,
        selected_index=payload.selectedIndex,
        duration_sec=payload.durationSec,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found.")

    question = next(
        (q for q in updated["questions"] if q["position"] == payload.position), None
    )
    return {
        "position": payload.position,
        "isCorrect": question["isCorrect"] if question else None,
        "status": updated["status"],
    }


@router.patch("/{session_id}/explanation")
async def record_explanation(
    session_id: str,
    courseId: str,
    payload: SessionExplanationRequest,
):
    """
    Record the student's explanation for a question.
    Triggers evaluation immediately and writes the signal to the session.
    """
    session = await session_service.get_session(session_id, courseId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    question = next(
        (q for q in session["questions"] if q["position"] == payload.position), None
    )
    if not question:
        raise HTTPException(status_code=404, detail=f"Question {payload.position} not found.")

    # Save explanation
    await session_service.record_explanation(
        session_id=session_id,
        course_id=courseId,
        position=payload.position,
        student_explanation=payload.studentExplanation,
    )

    # Trigger evaluation
    try:
        signal = await evaluate_reasoning(
            question=question["question"],
            options=question["options"],
            correct_index=question["correctIndex"],
            selected_index=question.get("selectedIndex", 0),
            student_explanation=payload.studentExplanation,
        )
        await session_service.record_evaluation(
            session_id=session_id,
            course_id=courseId,
            position=payload.position,
            signal=signal.signal,
            confidence=signal.confidence,
            faculty_insight=signal.facultyInsight,
            student_feedback=signal.studentFeedback,
        )
        return {
            "position": payload.position,
            "signal": signal.signal,
            "confidence": signal.confidence,
            "studentFeedback": signal.studentFeedback,
            "facultyInsight": signal.facultyInsight,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.patch("/{session_id}/chat")
async def append_chat(session_id: str, courseId: str, payload: SessionChatRequest):
    """Append a single chat message to the session history."""
    updated = await session_service.append_chat(
        session_id=session_id,
        course_id=courseId,
        role=payload.role,
        text=payload.text,
        question_position=payload.questionPosition,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "ok", "chatLength": len(updated.get("chatHistory", []))}


@router.post("/{session_id}/complete")
async def complete_session(session_id: str, courseId: str):
    """
    Finalise the session — compute summary, set completedAt, status: completed.
    Called when the student views the results screen.
    """
    updated = await session_service.complete_session(session_id, courseId)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "status": "completed",
        "summary": updated.get("summary"),
        "completedAt": updated.get("completedAt"),
    }
