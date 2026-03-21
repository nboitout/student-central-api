from fastapi import APIRouter, HTTPException
from models.course import Course, CourseCreate, CourseUpdate
from services import cosmos_service
from services.blob_service import delete_blob

router = APIRouter(prefix="/api/courses", tags=["courses"])


@router.get("")
async def list_courses(userId: str = "nicolas"):
    """Return all courses for a user."""
    courses = await cosmos_service.list_courses(user_id=userId)
    return {"courses": courses, "count": len(courses)}


@router.post("", status_code=201)
async def create_course(payload: CourseCreate):
    """Create a new course card."""
    course = Course(
        title=payload.title,
        author=payload.author,
        source=payload.source,
        userId=payload.userId,
        exercisesTotal=payload.exercisesTotal,
    )
    created = await cosmos_service.create_course(course)
    return created


@router.get("/{course_id}")
async def get_course(course_id: str, userId: str = "nicolas"):
    """Get a single course by ID."""
    course = await cosmos_service.get_course(course_id, user_id=userId)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found.")
    return course


@router.patch("/{course_id}")
async def update_course(course_id: str, updates: CourseUpdate, userId: str = "nicolas"):
    """Update course status or progress."""
    updated = await cosmos_service.update_course(course_id, updates, user_id=userId)
    if not updated:
        raise HTTPException(status_code=404, detail="Course not found.")
    return updated


@router.delete("/{course_id}", status_code=204)
async def delete_course(course_id: str, userId: str = "nicolas"):
    """Delete a course and its associated PDF from Blob Storage."""
    # Fetch course first to get the PDF URL
    course = await cosmos_service.get_course(course_id, user_id=userId)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found.")

    # Delete from Cosmos DB
    deleted = await cosmos_service.delete_course(course_id, user_id=userId)
    if not deleted:
        raise HTTPException(status_code=404, detail="Course not found.")

    # Also delete the PDF blob if it exists
    if course.get("pdfUrl"):
        await delete_blob(course["pdfUrl"])

    return None


@router.patch("/{course_id}/pdf")
async def attach_pdf(course_id: str, pdfUrl: str, userId: str = "nicolas"):
    """
    Attach a blob URL to an existing course after upload.
    Called after POST /api/upload succeeds.
    """
    course = await cosmos_service.get_course(course_id, user_id=userId)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found.")

    course["pdfUrl"] = pdfUrl
    course["source"] = pdfUrl.split("/")[-1].split("?")[0]

    from azure.cosmos import CosmosClient
    import os
    client = CosmosClient(
        os.getenv("AZURE_COSMOS_ENDPOINT"),
        credential=os.getenv("AZURE_COSMOS_KEY")
    )
    db = client.get_database_client(os.getenv("AZURE_COSMOS_DATABASE", "student-central"))
    container = db.get_container_client(os.getenv("AZURE_COSMOS_CONTAINER", "courses"))
    container.replace_item(item=course_id, body=course)

    return course
