import httpx
import fitz  # PyMuPDF
import base64
from typing import List


async def render_pdf_pages_as_images(
    sas_url: str,
    max_pages: int = 10,
    dpi: int = 150,
) -> List[str]:
    """
    Download a PDF from a SAS URL and render each page as a base64-encoded PNG.
    Returns a list of base64 strings (one per page, up to max_pages).

    dpi=150 gives good quality while keeping image sizes reasonable for the API.
    Higher DPI = better quality but more tokens. 150 is the sweet spot.
    """
    # Download PDF bytes
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(sas_url)
        response.raise_for_status()
        pdf_bytes = response.content

    # Open PDF with PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    pages_to_render = min(total_pages, max_pages)

    images = []
    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pages_to_render):
        page = doc[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        images.append(b64)

    doc.close()
    return images
