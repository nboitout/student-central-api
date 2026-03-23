import os
from azure.cosmos import CosmosClient, exceptions
from models.course import Course, CourseUpdate
from models.mcq import StoredMCQ
from datetime import datetime


def get_cosmos_client():
    endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
    key = os.getenv("AZURE_COSMOS_KEY")
    if not endpoint or not key:
        raise ValueError("Cosmos DB credentials are not set")
    return CosmosClient(endpoint, credential=key)


def get_container(container_name: str):
    client = get_cosmos_client()
    db_name = os.getenv("AZURE_COSMOS_DATABASE", "student-central")
    database = client.get_database_client(db_name)
    return database.get_container_client(container_name)


def get_courses_container():
    return get_container(os.getenv("AZURE_COSMOS_CONTAINER", "courses"))


def get_mcqs_container():
    return get_container("mcqs")


# ── Courses ───────────────────────────────────────────────

async def create_course(course: Course) -> dict:
    container = get_courses_container()
    item = course.model_dump()
    container.create_item(body=item)
    return item


async def list_courses(user_id: str = "nicolas") -> list[dict]:
    container = get_courses_container()
    query = "SELECT * FROM c WHERE c.userId = @userId ORDER BY c.createdAt DESC"
    params = [{"name": "@userId", "value": user_id}]
    return list(container.query_items(
        query=query, parameters=params, enable_cross_partition_query=True
    ))


async def get_course(course_id: str, user_id: str = "nicolas") -> dict | None:
    container = get_courses_container()
    try:
        return container.read_item(item=course_id, partition_key=user_id)
    except exceptions.CosmosResourceNotFoundError:
        return None


async def update_course(course_id: str, updates: CourseUpdate, user_id: str = "nicolas") -> dict | None:
    container = get_courses_container()
    try:
        item = container.read_item(item=course_id, partition_key=user_id)
    except exceptions.CosmosResourceNotFoundError:
        return None

    if updates.status is not None:
        item["status"] = updates.status
    if updates.exercisesDone is not None:
        item["exercisesDone"] = updates.exercisesDone
    if updates.allowDownload is not None:
        item["allowDownload"] = updates.allowDownload
    if updates.mcqStatus is not None:
        item["mcqStatus"] = updates.mcqStatus
    if updates.mcqCount is not None:
        item["mcqCount"] = updates.mcqCount

    item["updatedAt"] = datetime.utcnow().isoformat()
    container.replace_item(item=course_id, body=item)
    return item


async def delete_course(course_id: str, user_id: str = "nicolas") -> bool:
    container = get_courses_container()
    try:
        container.delete_item(item=course_id, partition_key=user_id)
        return True
    except exceptions.CosmosResourceNotFoundError:
        return False


async def update_course_raw(course_id: str, item: dict) -> dict:
    """Replace an entire course document — used internally."""
    container = get_courses_container()
    item["updatedAt"] = datetime.utcnow().isoformat()
    container.replace_item(item=course_id, body=item)
    return item


# ── MCQ Bank ──────────────────────────────────────────────

async def save_mcq_bank(mcqs: list[StoredMCQ]) -> int:
    """
    Store a list of MCQ questions in the mcqs container.
    Returns the number of questions saved.
    """
    container = get_mcqs_container()
    saved = 0
    for mcq in mcqs:
        container.create_item(body=mcq.model_dump())
        saved += 1
    return saved


async def get_mcq_bank(course_id: str) -> list[dict]:
    """Return all stored MCQs for a given course."""
    container = get_mcqs_container()
    query = "SELECT * FROM c WHERE c.courseId = @courseId"
    params = [{"name": "@courseId", "value": course_id}]
    return list(container.query_items(
        query=query, parameters=params, enable_cross_partition_query=True
    ))


async def delete_mcq_bank(course_id: str) -> int:
    """Delete all MCQs for a course — called when course is deleted."""
    container = get_mcqs_container()
    query = "SELECT c.id, c.courseId FROM c WHERE c.courseId = @courseId"
    params = [{"name": "@courseId", "value": course_id}]
    items = list(container.query_items(
        query=query, parameters=params, enable_cross_partition_query=True
    ))
    deleted = 0
    for item in items:
        try:
            container.delete_item(item=item["id"], partition_key=item["courseId"])
            deleted += 1
        except Exception:
            pass
    return deleted
