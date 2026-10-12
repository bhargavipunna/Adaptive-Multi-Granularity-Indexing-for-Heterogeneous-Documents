"""
AMHIE: Adaptive Multi-Granularity Hierarchical Indexing Engine
Main FastAPI Application Entrypoint.
Modular REST API with live storage inspection, C++ N-ary tree analytics,
adaptive RAG query console, and static frontend hosting.
"""

import os
import sys
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Ensure python package is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.services.service_container import get_service_container
from api.routers import documents_router, tree_router, storage_router, rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the service container and pre-indexes the benchmark sample."""
    print("\n[AMHIE Server] Initializing AMHIE Service Engine...")
    container = get_service_container()
    try:
        sample_doc = container.ingest_sample_document(threshold=0.45)
        print(f"[AMHIE Server] Preloaded '{sample_doc['filename']}' ({sample_doc['total_pages']} pages, {sample_doc['total_nodes_in_tree']} logical nodes, {sample_doc['total_vectors_indexed']} vectors).")
    except Exception as e:
        print(f"[AMHIE Server] Note: Sample preload skipped or deferred: {e}")
    print("[AMHIE Server] Backend services ready.\n")
    yield
    print("[AMHIE Server] Shutting down AMHIE services.")


app = FastAPI(
    title="AMHIE API",
    description="Adaptive Multi-Granularity Hierarchical Indexing Engine REST Service with Multi-Tier Storage Inspection",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Modular Routers
app.include_router(documents_router)
app.include_router(tree_router)
app.include_router(storage_router)
app.include_router(rag_router)

# Health & System Status
@app.get("/health")
def health() -> Dict[str, Any]:
    container = get_service_container()
    return {
        "status": "healthy",
        "engine": "AMHIE",
        "version": "2.0.0",
        "tree_engine": "C++17 libtree_engine",
        "vector_db": "Qdrant HNSW (m=16, ef=100)",
        "active_document": container.active_doc_id,
        "total_documents": len(container.document_metadata)
    }


class AMHIEService:
    """Backward compatibility wrapper delegating to ServiceContainer."""
    def __init__(self):
        self._container = get_service_container()
        self.parser = self._container.parser
        self.feature_extractor = self._container.feature_extractor
        self.scorer = self._container.scorer
        self.vector_store = self._container.vector_store
        self.retriever = self._container.retriever
        self.rag_pipeline = self._container.rag_pipeline
        self.active_trees = self._container.active_trees
        self.document_stats = self._container.document_metadata

    def ingest_document(self, filename: str, content_bytes: bytes) -> Dict[str, Any]:
        res = self._container.ingest_document(filename, content_bytes)
        tree = self._container.active_trees.get(res["doc_id"])
        res["tree_json"] = tree.to_dict() if tree else {}
        return res

    def query(self, query_text: str) -> Dict[str, Any]:
        return self._container.query(query_text)

    def get_tree(self, doc_id: str) -> Optional[Dict[str, Any]]:
        tree = self._container.active_trees.get(doc_id)
        return tree.to_dict() if tree else None


# ==============================================================================
# Legacy Endpoints for Backward Compatibility
# ==============================================================================
class LegacyQueryModel(BaseModel):
    query: str
    top_k: Optional[int] = 3


@app.post("/upload_pdf")
async def legacy_upload_pdf(file: UploadFile = File(...)):
    container = get_service_container()
    try:
        content = await file.read()
        res = container.ingest_document(file.filename, content)
        tree = container.active_trees.get(res["doc_id"])
        res["tree_json"] = tree.to_dict() if tree else {}
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_query")
def legacy_search_query(req: LegacyQueryModel):
    container = get_service_container()
    try:
        res = container.query(req.query)
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tree_structure/{doc_id}")
def legacy_tree_structure(doc_id: str):
    container = get_service_container()
    tree = container.active_trees.get(doc_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Document tree not found")
    return JSONResponse(content=tree.to_dict())


# ==============================================================================
# Modern Motion Frontend Serving
# ==============================================================================
static_dir = os.path.join(project_root, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
@app.get("/app", include_in_schema=False)
def serve_frontend():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AMHIE API v2.0 is running. Visit /docs for OpenAPI specifications."}


if __name__ == "__main__":
    import uvicorn
    print("[AMHIE] Starting FastAPI server on port 8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
