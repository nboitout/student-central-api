from fastapi import APIRouter, UploadFile, File, HTTPException
from services.blob_service import upload_pdf

router = APIRouter(prefix="/api/upload", tags=["upload"])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("")
async def upload_course_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file to Azure Blob Storage.
    Returns the permanent blob URL to store with the course record.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    # Upload to Azure Blob Storage
    blob_url = await upload_pdf(
        file_bytes=contents,
        original_filename=file.filename,
    )

    return {
        "url": blob_url,
        "filename": file.filename,
        "size": len(contents),
    }
