from fastapi import APIRouter, HTTPException
from models.session import (
    SessionCreateRequest,
    SessionAnswerRequest,
    SessionExplanationRequest,
    SessionChatRequest,
)
from services import session_service
from services.openai_service import evaluate_reasoning
from services.blob_service import get_blob_sas_url

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _get_sas_url(raw_url: str | None) -> str | None:
    """Convert a private blob URL to a 2-hour SAS URL. Returns None on failure."""
    if not raw_url:
        return None
    try:
        return get_blob_sas_url(raw_url, expiry_hours=2)
    except Exception:
        return None


def _build_question_payload(session_q: dict, mcq_bank_item: dict | None = None) -> dict:
    """
    Build the question payload sent to the frontend.
    Priority for slideImageUrl:
      1. mcq_bank_item (freshest source — has slideImageUrl from generation time)
      2. session_q dict (stored at session creation — may be null for old sessions)
    Converts private blob URL to 2-hour SAS URL.
    Uses snake_case keys to match frontend expectations.
    """
    # Resolve slideImageUrl — prefer MCQ bank over session record
    raw_slide_url = (
        (mcq_bank_item or {}).get("slideImageUrl")
        or session_q.get("slideImageUrl")
        or session_q.get("slide_image_url")
    )

    return {
        "position":        session_q.get("position"),
        "mcqId":           session_q.get("mcqId"),
        "question":        session_q.get("question"),
        "options":         session_q.get("options"),
        "correctIndex":    session_q.get("correctIndex"),
        "page_number":     (
            (mcq_bank_item or {}).get("pageNumber")
            or session_q.get("pageNumber")
            or session_q.get("page_number")
        ),
        "slide_image_url": _get_sas_url(raw_slide_url),
    }


async def _fetch_mcq_bank_item(mcq_id: str, course_id: str) -> dict | None:
    """Fetch a single MCQ from the bank by ID."""
    if not mcq_id:
        return None
    try:
        from services.cosmos_service import get_mcqs_container
        container = get_mcqs_container()
        return container.read_item(item=mcq_id, partition_key=course_id)
    except Exception:
        return None


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

    first_q = session.questions[0].model_dump()
    first_drawn = drawn[0] if drawn else None

    return {
        "sessionId":      session.id,
        "mode":           session.mode,
        "language":       session.language,
        "totalQuestions": len(session.questions),
        "question":       _build_question_payload(first_q, first_drawn),
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
    Return the question at a given position with a fresh SAS URL for the slide.
    Called when the frontend advances to Q2, Q3, Q4, Q5.
    Always fetches slideImageUrl from the MCQ bank to ensure freshness.
    """
    session = await session_service.get_session(session_id, courseId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    session_q = next(
        (q for q in session["questions"] if q["position"] == position), None
    )
    if not session_q:
        raise HTTPException(status_code=404, detail=f"Question {position} not found.")

    # Always fetch from MCQ bank for freshest slideImageUrl
    mcq_bank_item = await _fetch_mcq_bank_item(
        mcq_id=session_q.get("mcqId"),
        course_id=courseId,
    )

    return _build_question_payload(session_q, mcq_bank_item)


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
        "position":  payload.position,
        "isCorrect": question["isCorrect"] if question else None,
        "status":    updated["status"],
    }


@router.patch("/{session_id}/explanation")
async def record_explanation(
    session_id: str,
    courseId: str,
    payload: SessionExplanationRequest,
):
    """
    Record the student's explanation and trigger evaluation immediately.
    Returns the evaluation signal in the response.
    """
    session = await session_service.get_session(session_id, courseId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    question = next(
        (q for q in session["questions"] if q["position"] == payload.position), None
    )
    if not question:
        raise HTTPException(status_code=404, detail=f"Question {payload.position} not found.")

    await session_service.record_explanation(
        session_id=session_id,
        course_id=courseId,
        position=payload.position,
        student_explanation=payload.studentExplanation,
    )

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
            "position":        payload.position,
            "signal":          signal.signal,
            "confidence":      signal.confidence,
            "studentFeedback": signal.studentFeedback,
            "facultyInsight":  signal.facultyInsight,
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
        "status":      "completed",
        "summary":     updated.get("summary"),
        "completedAt": updated.get("completedAt"),
    }
