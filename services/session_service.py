import os
import random
from azure.cosmos import CosmosClient, exceptions
from datetime import datetime
from models.session import Session, QuestionRecord, ChatEntry, SessionSummary


def get_cosmos_client():
    endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
    key = os.getenv("AZURE_COSMOS_KEY")
    if not endpoint or not key:
        raise ValueError("Cosmos DB credentials are not set")
    return CosmosClient(endpoint, credential=key)


def get_sessions_container():
    client = get_cosmos_client()
    db = client.get_database_client(os.getenv("AZURE_COSMOS_DATABASE", "student-central"))
    return db.get_container_client("sessions")


def get_mcqs_container():
    client = get_cosmos_client()
    db = client.get_database_client(os.getenv("AZURE_COSMOS_DATABASE", "student-central"))
    return db.get_container_client("mcqs")


SESSION_SIZE = 5


async def get_previously_seen_mcq_ids(course_id: str, user_id: str) -> set[str]:
    """Return all mcqIds seen by this user in previous sessions for this course."""
    container = get_sessions_container()
    query = (
        "SELECT VALUE q.mcqId FROM s JOIN q IN s.questions "
        "WHERE s.courseId = @courseId AND s.userId = @userId AND s.status = 'completed'"
    )
    params = [
        {"name": "@courseId", "value": course_id},
        {"name": "@userId",   "value": user_id},
    ]
    results = list(container.query_items(
        query=query, parameters=params, enable_cross_partition_query=True
    ))
    return set(results)


async def draw_questions(course_id: str, user_id: str) -> list[dict]:
    """
    Draw SESSION_SIZE questions from the bank, excluding previously seen IDs.
    Falls back to full bank if not enough unseen questions remain.
    """
    container = get_mcqs_container()
    query = "SELECT * FROM c WHERE c.courseId = @courseId"
    params = [{"name": "@courseId", "value": course_id}]
    bank = list(container.query_items(
        query=query, parameters=params, enable_cross_partition_query=True
    ))

    if not bank:
        return []

    seen_ids = await get_previously_seen_mcq_ids(course_id, user_id)
    unseen = [q for q in bank if q["id"] not in seen_ids]

    # Fall back to full bank if not enough unseen questions
    pool = unseen if len(unseen) >= SESSION_SIZE else bank

    return random.sample(pool, min(SESSION_SIZE, len(pool)))


async def create_session(
    course_id: str,
    user_id: str,
    mode: str,
    language: str,
) -> tuple[Session, list[dict]]:
    """
    Create a new session and pre-select SESSION_SIZE questions.
    Returns the session and the raw MCQ list (for serving to the frontend).
    """
    drawn = await draw_questions(course_id, user_id)
    if not drawn:
        raise ValueError("No questions available in the MCQ bank for this course.")

    questions = [
        QuestionRecord(
            position=i + 1,
            mcqId=q["id"],
            question=q["question"],
            options=[opt["text"] for opt in q["options"]],
            correctIndex=q["correctIndex"],
            pageNumber=q.get("pageNumber"),
        )
        for i, q in enumerate(drawn)
    ]

    session = Session(
        courseId=course_id,
        userId=user_id,
        mode=mode,
        language=language,
        status="started",
        questions=questions,
    )

    container = get_sessions_container()
    container.create_item(body=session.model_dump())

    return session, drawn


async def get_session(session_id: str, course_id: str) -> dict | None:
    container = get_sessions_container()
    try:
        return container.read_item(item=session_id, partition_key=course_id)
    except exceptions.CosmosResourceNotFoundError:
        return None


async def patch_session(session_id: str, course_id: str, updates: dict) -> dict | None:
    """Apply a partial update to a session document."""
    container = get_sessions_container()
    try:
        item = container.read_item(item=session_id, partition_key=course_id)
    except exceptions.CosmosResourceNotFoundError:
        return None

    for key, value in updates.items():
        item[key] = value

    item["updatedAt"] = datetime.utcnow().isoformat()
    container.replace_item(item=session_id, body=item)
    return item


async def record_answer(
    session_id: str,
    course_id: str,
    position: int,
    selected_index: int,
    duration_sec: int,
) -> dict | None:
    item = await get_session(session_id, course_id)
    if not item:
        return None

    for q in item["questions"]:
        if q["position"] == position:
            q["selectedIndex"]  = selected_index
            q["isCorrect"]      = selected_index == q["correctIndex"]
            q["durationSec"]    = duration_sec
            q["answeredAt"]     = datetime.utcnow().isoformat()
            break

    item["status"] = "answering"
    return await patch_session(session_id, course_id, {
        "questions": item["questions"],
        "status": item["status"],
    })


async def record_explanation(
    session_id: str,
    course_id: str,
    position: int,
    student_explanation: str,
) -> dict | None:
    item = await get_session(session_id, course_id)
    if not item:
        return None

    for q in item["questions"]:
        if q["position"] == position:
            q["studentExplanation"] = student_explanation
            break

    return await patch_session(session_id, course_id, {"questions": item["questions"]})


async def record_evaluation(
    session_id: str,
    course_id: str,
    position: int,
    signal: str,
    confidence: str,
    faculty_insight: str,
    student_feedback: str,
) -> dict | None:
    item = await get_session(session_id, course_id)
    if not item:
        return None

    for q in item["questions"]:
        if q["position"] == position:
            q["evaluationSignal"]     = signal
            q["evaluationConfidence"] = confidence
            q["facultyInsight"]       = faculty_insight
            q["studentFeedback"]      = student_feedback
            break

    return await patch_session(session_id, course_id, {"questions": item["questions"]})


async def append_chat(
    session_id: str,
    course_id: str,
    role: str,
    text: str,
    question_position: int | None,
) -> dict | None:
    item = await get_session(session_id, course_id)
    if not item:
        return None

    entry = ChatEntry(
        role=role,
        text=text,
        questionPosition=question_position,
    ).model_dump()

    chat = item.get("chatHistory", [])
    chat.append(entry)

    return await patch_session(session_id, course_id, {
        "chatHistory": chat,
        "status": "chatting",
    })


async def complete_session(session_id: str, course_id: str) -> dict | None:
    item = await get_session(session_id, course_id)
    if not item:
        return None

    questions = item.get("questions", [])
    correct_count = sum(1 for q in questions if q.get("isCorrect"))
    total_duration = sum(q.get("durationSec", 0) for q in questions)

    signal_breakdown = {
        "Strong": 0,
        "Fragile": 0,
        "Partial misconception": 0,
        "Low mastery": 0,
        "unevaluated": 0,
    }
    for q in questions:
        sig = q.get("evaluationSignal")
        if sig and sig in signal_breakdown:
            signal_breakdown[sig] += 1
        else:
            signal_breakdown["unevaluated"] += 1

    summary = SessionSummary(
        totalQuestions=len(questions),
        correctCount=correct_count,
        totalDurationSec=total_duration,
        signalBreakdown=signal_breakdown,
    ).model_dump()

    return await patch_session(session_id, course_id, {
        "status": "completed",
        "completedAt": datetime.utcnow().isoformat(),
        "summary": summary,
    })


async def list_sessions(course_id: str, user_id: str) -> list[dict]:
    container = get_sessions_container()
    query = (
        "SELECT * FROM c WHERE c.courseId = @courseId AND c.userId = @userId "
        "ORDER BY c.startedAt DESC"
    )
    params = [
        {"name": "@courseId", "value": course_id},
        {"name": "@userId",   "value": user_id},
    ]
    return list(container.query_items(
        query=query, parameters=params, enable_cross_partition_query=True
    ))
