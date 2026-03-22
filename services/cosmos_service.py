import os
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from models.course import Course, CourseUpdate
from datetime import datetime


def get_cosmos_client():
    endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
    key = os.getenv("AZURE_COSMOS_KEY")
    if not endpoint or not key:
        raise ValueError("Cosmos DB credentials are not set")
    return CosmosClient(endpoint, credential=key)


def get_container():
    client = get_cosmos_client()
    db_name = os.getenv("AZURE_COSMOS_DATABASE", "student-central")
    container_name = os.getenv("AZURE_COSMOS_CONTAINER", "courses")
    database = client.get_database_client(db_name)
    return database.get_container_client(container_name)


async def create_course(course: Course) -> dict:
    """Insert a new course document into Cosmos DB."""
    container = get_container()
    item = course.model_dump()
    container.create_item(body=item)
    return item


async def list_courses(user_id: str = "nicolas") -> list[dict]:
    """Return all courses for a given user, ordered by createdAt descending."""
    container = get_container()
    query = (
        "SELECT * FROM c WHERE c.userId = @userId "
        "ORDER BY c.createdAt DESC"
    )
    params = [{"name": "@userId", "value": user_id}]
    items = list(container.query_items(
        query=query,
        parameters=params,
        enable_cross_partition_query=True,
    ))
    return items


async def get_course(course_id: str, user_id: str = "nicolas") -> dict | None:
    """Fetch a single course by ID."""
    container = get_container()
    try:
        item = container.read_item(item=course_id, partition_key=user_id)
        return item
    except exceptions.CosmosResourceNotFoundError:
        return None


async def update_course(course_id: str, updates: CourseUpdate, user_id: str = "nicolas") -> dict | None:
    """Patch a course with provided fields."""
    container = get_container()
    try:
        item = container.read_item(item=course_id, partition_key=user_id)
    except exceptions.CosmosResourceNotFoundError:
        return None

    if updates.status is not None:
        item["status"] = updates.status
    if updates.exercisesDone is not None:
        item["exercisesDone"] = updates.exercisesDone

    item["updatedAt"] = datetime.utcnow().isoformat()
    container.replace_item(item=course_id, body=item)
    return item


async def delete_course(course_id: str, user_id: str = "nicolas") -> bool:
    """Delete a course document."""
    container = get_container()
    try:
        container.delete_item(item=course_id, partition_key=user_id)
        return True
    except exceptions.CosmosResourceNotFoundError:
        return False
