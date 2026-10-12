"""
Storage Inspection Router for AMHIE.
Exposes endpoints demonstrating how data is structured and stored across
C++ memory, Qdrant HNSW vector index, multimodal parse tree, and feature metrics.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List

from ..services.service_container import get_service_container
from ..services.storage_service import StorageInspectorService

router = APIRouter(prefix="/api/storage", tags=["Storage & Architecture Inspection"])


@router.get("/overview")
def get_storage_overview() -> Dict[str, Any]:
    """Provides high-level system storage telemetry across all 4 tiers."""
    container = get_service_container()
    service = StorageInspectorService(container)
    return service.get_storage_overview()


@router.get("/vectors")
def get_vector_records(limit: int = Query(default=100, ge=1, le=500)) -> List[Dict[str, Any]]:
    """Inspects live vector database points, payloads, and embeddings in Qdrant."""
    container = get_service_container()
    service = StorageInspectorService(container)
    return service.get_all_vector_points(limit=limit)


@router.get("/hnsw")
def get_hnsw_specifications() -> Dict[str, Any]:
    """Exposes HNSW vector graph indexing parameters, metrics, and theoretical guarantees."""
    container = get_service_container()
    service = StorageInspectorService(container)
    return service.get_hnsw_architecture()


@router.get("/layers")
def get_storage_layers() -> Dict[str, Any]:
    """Returns technical architectural blueprints for the 4 storage tiers."""
    container = get_service_container()
    service = StorageInspectorService(container)
    return service.get_storage_layers_architecture()


@router.get("/document/{doc_id}")
def get_document_storage_breakdown(doc_id: str) -> Dict[str, Any]:
    """Returns detailed storage breakdown for an individual ingested document."""
    container = get_service_container()
    doc_meta = container.document_metadata.get(doc_id)
    if not doc_meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in storage registry.")

    tree = container.active_trees.get(doc_id)
    nodes = tree.get_all_nodes() if tree else []

    return {
        "doc_id": doc_id,
        "metadata": doc_meta,
        "total_nodes_in_memory": len(nodes),
        "node_type_distribution": {
            "DOCUMENT": sum(1 for n in nodes if n.get("node_type") == "DOCUMENT"),
            "PAGE_COARSE": sum(1 for n in nodes if n.get("node_type") == "PAGE_COARSE"),
            "PAGE_COMPLEX": sum(1 for n in nodes if n.get("node_type") == "PAGE_COMPLEX"),
            "TABLE": sum(1 for n in nodes if n.get("node_type") == "TABLE"),
            "IMAGE": sum(1 for n in nodes if n.get("node_type") == "IMAGE"),
            "TEXT": sum(1 for n in nodes if n.get("node_type") == "TEXT"),
        },
        "sample_nodes": nodes[:10]
    }
