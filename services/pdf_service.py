import httpx
import fitz  # PyMuPDF
import base64
from typing import List, Tuple
from azure.storage.blob import BlobServiceClient, ContentSettings
import os


def get_blob_client() -> BlobServiceClient:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")
    return BlobServiceClient.from_connection_string(connection_string)


async def render_pdf_pages_as_images(
    sas_url: str,
    max_pages: int = 10,
    dpi: int = 150,
) -> List[str]:
    """
    Download PDF and render pages as base64 PNG strings.
    Used for passing to GPT-5.2-chat.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(sas_url)
        response.raise_for_status()
        pdf_bytes = response.content

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = min(len(doc), max_pages)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    images = []
    for page_num in range(total_pages):
        page = doc[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")
        images.append(base64.b64encode(png_bytes).decode("utf-8"))

    doc.close()
    return images


async def render_and_store_pdf_pages(
    sas_url: str,
    course_id: str,
    max_pages: int = 10,
    dpi: int = 150,
) -> List[Tuple[int, str]]:
    """
    Download PDF, render each page as PNG, upload each to Blob Storage.
    Returns list of (page_number, blob_url) tuples.
    Called once at MCQ generation time — persists slide images for later display.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(sas_url)
        response.raise_for_status()
        pdf_bytes = response.content

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = min(len(doc), max_pages)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    blob_client = get_blob_client()
    container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "course-pdfs")

    page_urls = []
    for page_num in range(total_pages):
        page = doc[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")

        blob_name = f"slides/{course_id}/page_{page_num}.png"
        bc = blob_client.get_blob_client(container=container, blob=blob_name)
        bc.upload_blob(
            png_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type="image/png")
        )
        page_urls.append((page_num, bc.url))

    doc.close()
    return page_urls
