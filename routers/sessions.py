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


def _build_question_payload(q, drawn_mcq: dict | None = None) -> dict:
    """
    Build the question payload sent to the frontend.
    Converts the private slideImageUrl blob URL to a 2-hour SAS URL.
    Uses snake_case keys to match frontend expectations.
    """
    # Get slideImageUrl from either the QuestionRecord or the raw drawn MCQ
    slide_image_url = None
    page_number = q.pageNumber if hasattr(q, "pageNumber") else q.get("pageNumber")

    raw_slide_url = None
    if hasattr(q, "pageNumber"):
        # QuestionRecord object — slideImageUrl not stored on it directly
        # get it from the drawn MCQ dict if available
        if drawn_mcq:
            raw_slide_url = drawn_mcq.get("slideImageUrl")
    else:
        # dict from Cosmos DB
        raw_slide_url = q.get("slideImageUrl")

    if raw_slide_url:
        try:
            slide_image_url = get_blob_sas_url(raw_slide_url, expiry_hours=2)
        except Exception:
            slide_image_url = None

    position = q.position if hasattr(q, "position") else q.get("position")
    mcq_id   = q.mcqId    if hasattr(q, "mcqId")    else q.get("mcqId")
    question = q.question  if hasattr(q, "question")  else q.get("question")
    options  = q.options   if hasattr(q, "options")   else q.get("options")
    correct  = q.correctIndex if hasattr(q, "correctIndex") else q.get("correctIndex")

    return {
        "position":        position,
        "mcqId":           mcq_id,
        "question":        question,
        "options":         options,
        "correctIndex":    correct,
        "page_number":     page_number,
        "slide_image_url": slide_image_url,
    }


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
    # drawn[0] is the raw MCQ dict with slideImageUrl
    first_drawn = drawn[0] if drawn else None

    return {
        "sessionId":      session.id,
        "mode":           session.mode,
        "language":       session.language,
        "totalQuestions": len(session.questions),
        "question":       _build_question_payload(first, first_drawn),
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
    """
    session = await session_service.get_session(session_id, courseId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    question = next(
        (q for q in session["questions"] if q["position"] == position), None
    )
    if not question:
        raise HTTPException(status_code=404, detail=f"Question {position} not found.")

    # Fetch the raw MCQ from the bank to get slideImageUrl
    from services.cosmos_service import get_mcqs_container
    mcq_id = question.get("mcqId")
    drawn_mcq = None
    if mcq_id:
        try:
            container = get_mcqs_container()
            drawn_mcq = container.read_item(item=mcq_id, partition_key=courseId)
        except Exception:
            drawn_mcq = None

    return _build_question_payload(question, drawn_mcq)


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
