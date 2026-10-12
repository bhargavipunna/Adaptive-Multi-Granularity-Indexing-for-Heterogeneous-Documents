"""
Document Management Router for AMHIE.
Handles multi-format document ingestion, parsing, feature extraction,
and sample loading.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Form
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..services.service_container import get_service_container

router = APIRouter(prefix="/api/documents", tags=["Document Ingestion & Management"])


class ReindexRequest(BaseModel):
    threshold: Optional[float] = 0.45


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.45)
) -> Dict[str, Any]:
    """Uploads a PDF, TXT, or Markdown document, builds N-ary tree, and indexes into Qdrant."""
    container = get_service_container()
    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        result = container.ingest_document(file.filename, content_bytes, threshold=threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.post("/sample")
def load_sample_document(threshold: float = Query(default=0.45, ge=0.1, le=0.9)) -> Dict[str, Any]:
    """Instantly loads the heterogeneous enterprise financial benchmark document."""
    container = get_service_container()
    try:
        result = container.ingest_sample_document(threshold=threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample document: {str(e)}")


@router.get("/list")
def list_documents() -> List[Dict[str, Any]]:
    """Lists all ingested documents with complexity and vector statistics."""
    container = get_service_container()
    return list(container.document_metadata.values())


@router.get("/{doc_id}")
def get_document_details(doc_id: str) -> Dict[str, Any]:
    """Returns detailed statistics for a specific document."""
    container = get_service_container()
    meta = container.document_metadata.get(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return meta
