"""
Adaptive RAG Query & Traversal Router for AMHIE.
Handles multi-granularity query traversal, prompt formatting, and synthesis.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..services.service_container import get_service_container

router = APIRouter(prefix="/api/rag", tags=["Adaptive Retrieval & Generation"])


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


@router.post("/generate")
def generate_rag_response(req: QueryRequest) -> Dict[str, Any]:
    """
    Executes full AMHIE RAG pipeline:
    1. Coarse-first HNSW vector search.
    2. Adaptive branching into complex child subtrees.
    3. Context assembly and synthesized answer generation.
    """
    container = get_service_container()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        res = container.query(req.query)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Generation failed: {str(e)}")


@router.post("/search")
def adaptive_search(req: QueryRequest) -> Dict[str, Any]:
    """Executes adaptive two-stage retrieval only and returns context cards with traversal paths."""
    container = get_service_container()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        contexts = container.retriever.retrieve(req.query)
        return {
            "query": req.query,
            "total_retrieved": len(contexts),
            "contexts": [c.to_dict() for c in contexts],
            "traversal_paths": [c.traversal_path for c in contexts]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptive search failed: {str(e)}")


@router.get("/sample-queries")
def get_sample_queries() -> List[Dict[str, str]]:
    """Returns curated queries targeting both coarse narrative pages and fine-grained tables."""
    return [
        {
            "category": "Complex Multi-Row Financial Table",
            "query": "What were the Q4 2024 operating margins and revenue?",
            "expected_target": "Page 2 Financial Metrics Table"
        },
        {
            "category": "Capital Expenditure & Net Profit",
            "query": "How much was invested in R&D and what was net profit in Q4?",
            "expected_target": "Page 2 Financial Metrics Table"
        },
        {
            "category": "Hardware & Benchmarks Table",
            "query": "What is the Precision@5 and table preservation score of AMHIE compared to flat RAG?",
            "expected_target": "Page 3 Benchmark Comparison Table"
        },
        {
            "category": "Narrative System Overview (Coarse Page)",
            "query": "Why does uniform flat 512-token chunking fail on multimodal documents?",
            "expected_target": "Page 1 Introduction Coarse Node"
        },
        {
            "category": "Engineering Architecture (Coarse Page)",
            "query": "What are the HNSW graph indexing parameters and embedding model used?",
            "expected_target": "Page 4 Deployment Specifications Node"
        }
    ]
