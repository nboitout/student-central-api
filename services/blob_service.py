import os
import uuid
from azure.storage.blob import BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta, timezone


def get_blob_client() -> BlobServiceClient:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")
    return BlobServiceClient.from_connection_string(connection_string)


def get_container_name() -> str:
    return os.getenv("AZURE_STORAGE_CONTAINER_NAME", "course-pdfs")


async def upload_pdf(file_bytes: bytes, original_filename: str, user_id: str = "nicolas") -> str:
    """
    Upload a PDF to Azure Blob Storage.
    Returns the permanent blob URL.
    """
    client = get_blob_client()
    container = get_container_name()

    # Generate a unique blob name to avoid collisions
    ext = original_filename.rsplit(".", 1)[-1] if "." in original_filename else "pdf"
    blob_name = f"{user_id}/{uuid.uuid4()}.{ext}"

    blob_client = client.get_blob_client(container=container, blob=blob_name)
    blob_client.upload_blob(
        file_bytes,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/pdf")
    )

    return blob_client.url


def get_blob_sas_url(blob_url: str, expiry_hours: int = 2) -> str:
    """
    Generate a short-lived SAS URL for secure access to a private blob.
    Use this when passing the PDF URL to Azure OpenAI for reading.
    """
    client = get_blob_client()
    account_name = client.account_name
    account_key = client.credential.account_key
    container = get_container_name()

    # Extract blob name from URL
    blob_name = blob_url.split(f"{container}/")[-1]

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
    )

    return f"{blob_url}?{sas_token}"


async def delete_blob(blob_url: str) -> bool:
    """Delete a blob by its URL."""
    try:
        client = get_blob_client()
        container = get_container_name()
        blob_name = blob_url.split(f"{container}/")[-1].split("?")[0]
        blob_client = client.get_blob_client(container=container, blob=blob_name)
        blob_client.delete_blob()
        return True
    except Exception:
        return False
