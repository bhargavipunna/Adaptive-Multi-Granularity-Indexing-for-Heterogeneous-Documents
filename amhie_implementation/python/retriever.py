"""
Adaptive Hierarchical Retriever for AMHIE.
Executes two-stage adaptive HNSW tree traversal:
1. Search coarse parent nodes.
2. If a coarse page is flagged as high-complexity, fetch and rerank its fine-grained child elements.
3. If low-complexity, use coarse page text directly, avoiding micro-chunk overhead.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from .vector_store import AMHIEVectorStore


@dataclass
class RetrievedContext:
    node_id: str
    node_type: str
    page_number: int
    content: str
    similarity_score: float
    bounding_box: List[float]
    parent_id: Optional[str] = None
    traversal_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "page_number": self.page_number,
            "content": self.content,
            "similarity_score": round(self.similarity_score, 4),
            "bounding_box": self.bounding_box,
            "parent_id": self.parent_id,
            "traversal_path": self.traversal_path,
            "metadata": self.metadata
        }


class AdaptiveRetriever:
    """
    Orchestrates adaptive multi-granularity traversal over the HNSW vector database.
    """
    def __init__(self,
                 vector_store: AMHIEVectorStore,
                 top_k_coarse: int = 2,
                 top_k_fine: int = 4,
                 coarse_score_threshold: float = 0.15):
        self.vector_store = vector_store
        self.top_k_coarse = top_k_coarse
        self.top_k_fine = top_k_fine
        self.coarse_score_threshold = coarse_score_threshold

    def retrieve(self, query: str) -> List[RetrievedContext]:
        """
        Executes adaptive retrieval:
        1. Query HNSW vector index for top coarse page nodes.
        2. Examine node_type and complexity_score of each match.
        3. Branch into child nodes for PAGE_COMPLEX; retain PAGE_COARSE directly.
        """
        retrieved_contexts: List[RetrievedContext] = []

        # Step 1: Coarse search across page-level nodes
        coarse_results = self.vector_store.search_coarse_nodes(query, limit=self.top_k_coarse)

        for coarse_res in coarse_results:
            payload = coarse_res["payload"]
            score = coarse_res["score"]

            if score < self.coarse_score_threshold:
                continue

            node_id = payload.get("node_id")
            node_type = payload.get("node_type")
            page_num = payload.get("page_number", 0)
            has_children = payload.get("has_children", False) or node_type == "PAGE_COMPLEX"
            cs_score = payload.get("complexity_score", 0.0)

            # Step 2: Adaptive Branching
            if has_children:
                # High-complexity page: query fine-grained child elements
                child_results = self.vector_store.search_child_nodes(
                    parent_id=node_id,
                    query_text=query,
                    limit=self.top_k_fine
                )

                if child_results:
                    for child in child_results:
                        c_payload = child["payload"]
                        c_score = child["score"]

                        path_str = f"Root -> {node_id} (COMPLEX CS={cs_score:.2f}) -> {c_payload.get('node_id')} ({c_payload.get('node_type')})"
                        retrieved_contexts.append(RetrievedContext(
                            node_id=c_payload.get("node_id"),
                            node_type=c_payload.get("node_type"),
                            page_number=page_num,
                            content=c_payload.get("content", ""),
                            similarity_score=c_score,
                            bounding_box=c_payload.get("bounding_box", [0, 0, 0, 0]),
                            parent_id=node_id,
                            traversal_path=path_str,
                            metadata=c_payload.get("metadata", {})
                        ))
                else:
                    # Fallback to coarse node content if no children retrieved
                    path_str = f"Root -> {node_id} (COMPLEX Fallback)"
                    retrieved_contexts.append(RetrievedContext(
                        node_id=node_id,
                        node_type=node_type,
                        page_number=page_num,
                        content=payload.get("content", ""),
                        similarity_score=score,
                        bounding_box=payload.get("bounding_box", [0, 0, 0, 0]),
                        parent_id=payload.get("parent_id"),
                        traversal_path=path_str,
                        metadata=payload.get("metadata", {})
                    ))
            else:
                # Low-complexity page: retain coarse parent node directly
                path_str = f"Root -> {node_id} (COARSE CS={cs_score:.2f}) [Direct Retrieval]"
                retrieved_contexts.append(RetrievedContext(
                    node_id=node_id,
                    node_type=node_type,
                    page_number=page_num,
                    content=payload.get("content", ""),
                    similarity_score=score,
                    bounding_box=payload.get("bounding_box", [0, 0, 0, 0]),
                    parent_id=payload.get("parent_id"),
                    traversal_path=path_str,
                    metadata=payload.get("metadata", {})
                ))

        # Sort all retrieved contexts by similarity score
        retrieved_contexts.sort(key=lambda x: x.similarity_score, reverse=True)
        return retrieved_contexts
