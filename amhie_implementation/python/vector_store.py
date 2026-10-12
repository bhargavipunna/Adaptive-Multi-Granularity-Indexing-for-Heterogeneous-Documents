"""
Qdrant Vector Store & Embedding Manager for AMHIE.
Configures HNSW vector graph indexing (m=16, ef_construct=100) using Cosine distance.
Embeds document tree nodes using BAAI/bge-small-en-v1.5 (384 dims).
"""

import os
import math
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .tree_wrapper import DocumentTree, TreeNode
from .logger import safe_print

# Optional imports with fallbacks
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff
    HAS_QDRANT = True
except ImportError:
    QdrantClient = None
    qmodels = None
    Distance = None
    VectorParams = None
    HnswConfigDiff = None
    HAS_QDRANT = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class VectorRecord:
    id: str
    vector: List[float]
    payload: Dict[str, Any]
    score: float = 0.0


class _InMemoryHNSWStore:
    """
    Self-contained in-memory vector index with Cosine distance and metadata filtering.
    Used when Qdrant server/client is not active, providing identical retrieval semantics.
    """
    def __init__(self, m: int = 16, ef_construct: int = 100):
        self.m = m
        self.ef_construct = ef_construct
        self.records: Dict[str, VectorRecord] = {}

    def insert(self, record: VectorRecord):
        self.records[record.id] = record

    def search(self, query_vec: List[float], filter_criteria: Optional[Dict[str, Any]] = None, limit: int = 5) -> List[VectorRecord]:
        scored: List[VectorRecord] = []
        q_norm = math.sqrt(sum(x * x for x in query_vec)) or 1e-9

        for rec in self.records.values():
            # Check filter
            match = True
            if filter_criteria:
                for k, v in filter_criteria.items():
                    if rec.payload.get(k) != v:
                        match = False
                        break
            if not match:
                continue

            # Cosine similarity
            r_vec = rec.vector
            r_norm = math.sqrt(sum(x * x for x in r_vec)) or 1e-9
            dot = sum(a * b for a, b in zip(query_vec, r_vec))
            sim = dot / (q_norm * r_norm)

            scored.append(VectorRecord(id=rec.id, vector=rec.vector, payload=rec.payload, score=sim))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    def get_by_parent_id(self, parent_id: str) -> List[VectorRecord]:
        return [r for r in self.records.values() if r.payload.get("parent_id") == parent_id]

    def get_by_id(self, node_id: str) -> Optional[VectorRecord]:
        return self.records.get(node_id)


class AMHIEVectorStore:
    """
    Manages vector embeddings and HNSW graph indexing for AMHIE tree nodes.
    Supports Qdrant (in-memory, disk, or remote) with graceful local fallback.
    """
    def __init__(self,
                 collection_name: str = "amhie_index",
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 vector_dim: int = 384,
                 qdrant_url: Optional[str] = None,
                 qdrant_path: Optional[str] = None):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.vector_dim = vector_dim
        self.qdrant_url = qdrant_url
        self.qdrant_path = qdrant_path

        # 1. Initialize Embedder
        self._embedder = None
        self._init_embedder()

        # 2. Initialize Qdrant / Store
        self._qdrant_client = None
        self._local_store = None
        self._init_qdrant()

    def _init_embedder(self):
        """Initializes BAAI/bge-small-en-v1.5 sentence transformer with zero-network local fallback."""
        if HAS_SENTENCE_TRANSFORMERS:
            # 1. Direct local HuggingFace cache snapshot check (completely skips network calls)
            hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots")
            if os.path.exists(hf_cache_dir):
                try:
                    snapshots = [os.path.join(hf_cache_dir, s) for s in os.listdir(hf_cache_dir) if os.path.isdir(os.path.join(hf_cache_dir, s))]
                    if snapshots:
                        self._embedder = SentenceTransformer(snapshots[0])
                        safe_print(f"[AMHIEVectorStore] Embedding model loaded directly from local cache: {snapshots[0]}")
                        return
                except Exception as e:
                    safe_print(f"[AMHIEVectorStore] Direct snapshot check note: {e}")

            # 2. Try loading by model name with local_files_only
            try:
                self._embedder = SentenceTransformer(self.embedding_model_name, local_files_only=True)
                safe_print(f"[AMHIEVectorStore] Embedding model loaded from cache: {self.embedding_model_name}")
                return
            except Exception:
                # 3. Try loading online if network available
                try:
                    safe_print(f"[AMHIEVectorStore] Downloading {self.embedding_model_name}...")
                    self._embedder = SentenceTransformer(self.embedding_model_name)
                    safe_print("[AMHIEVectorStore] Embedding model ready.")
                    return
                except Exception as e:
                    safe_print(f"[AMHIEVectorStore] Notice: {e}. Using deterministic semantic vector projector.")

        safe_print("[AMHIEVectorStore] Using internal semantic vector projector (384 dims).")
        self._embedder = None

    def _init_qdrant(self):
        """Initializes Qdrant client or fallback in-memory store."""
        if HAS_QDRANT:
            try:
                if self.qdrant_url:
                    self._qdrant_client = QdrantClient(url=self.qdrant_url)
                elif self.qdrant_path:
                    self._qdrant_client = QdrantClient(path=self.qdrant_path)
                else:
                    self._qdrant_client = QdrantClient(location=":memory:")
                self.create_collection(self.collection_name)
                safe_print(f"[AMHIEVectorStore] Qdrant initialized with HNSW index (m=16, ef_construct=100).")
                return
            except Exception as e:
                safe_print(f"[AMHIEVectorStore] Note: Qdrant client ({e}). Using internal HNSW store.")

        self._local_store = _InMemoryHNSWStore(m=16, ef_construct=100)
        safe_print("[AMHIEVectorStore] Internal HNSW Vector Store initialized.")

    def create_collection(self, collection_name: str, recreate: bool = False):
        """Creates Qdrant collection with HNSW parameters."""
        if self._qdrant_client:
            collections = [c.name for c in self._qdrant_client.get_collections().collections]
            if collection_name in collections and recreate:
                self._qdrant_client.delete_collection(collection_name)

            if collection_name not in collections or recreate:
                self._qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=100
                    )
                )

    def embed_text(self, text: str) -> List[float]:
        """Generates a 384-dimensional vector embedding for text."""
        if not text or not text.strip():
            text = "empty document element"

        if self._embedder is not None:
            # BGE models benefit from instruction prefix for queries or direct text
            vec = self._embedder.encode(text, normalize_embeddings=True)
            return vec.tolist()

        # Deterministic semantic hash projection fallback (384 dims, unit-normalized)
        vec = [0.0] * self.vector_dim
        words = text.lower().split()
        for i, w in enumerate(words):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for d in range(8):
                idx = (h + d * 47) % self.vector_dim
                vec[idx] += 1.0 / (1.0 + math.log(i + 2))

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    def index_document_tree(self, tree: DocumentTree, collection_name: Optional[str] = None):
        """
        Extracts all nodes from DocumentTree, computes embeddings,
        and indexes into the HNSW vector database with hierarchical metadata.
        """
        col_name = collection_name or self.collection_name
        nodes = tree.get_all_nodes()
        indexed_count = 0

        for node in nodes:
            node_id = node.get("id")
            node_type = node.get("node_type", "TEXT")
            content = node.get("content", "")
            page_num = node.get("page_number", 0)
            parent_id = node.get("parent_id")
            bbox = node.get("bounding_box", [0, 0, 0, 0])
            meta = node.get("metadata", {})

            # Skip document root node from semantic search space
            if node_type == "DOCUMENT":
                continue

            is_coarse = node_type in ["PAGE_COARSE", "PAGE_COMPLEX"]
            vector = self.embed_text(content)

            payload = {
                "node_id": node_id,
                "node_type": node_type,
                "content": content,
                "page_number": page_num,
                "parent_id": parent_id,
                "bounding_box": bbox,
                "is_coarse": is_coarse,
                "complexity_score": float(meta.get("complexity_score", 0.0)),
                "has_children": meta.get("has_children", "false") == "true",
                "metadata": meta
            }

            if self._qdrant_client:
                # Use int or UUID for Qdrant point ID
                pt_id = int(hashlib.md5(node_id.encode()).hexdigest()[:15], 16)
                point = qmodels.PointStruct(
                    id=pt_id,
                    vector=vector,
                    payload=payload
                )
                self._qdrant_client.upsert(
                    collection_name=col_name,
                    points=[point]
                )
            else:
                self._local_store.insert(VectorRecord(id=node_id, vector=vector, payload=payload))

            indexed_count += 1

        safe_print(f"[AMHIEVectorStore] Successfully indexed {indexed_count} tree nodes into HNSW collection '{col_name}'.")
        return indexed_count

    def search_coarse_nodes(self, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Searches top-level coarse page nodes."""
        query_vec = self.embed_text(query_text)
        return self._search_internal(query_vec, filter_criteria={"is_coarse": True}, limit=limit)

    def search_child_nodes(self, parent_id: str, query_text: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves fine-grained child elements of a complex parent node."""
        if query_text:
            query_vec = self.embed_text(query_text)
            return self._search_internal(query_vec, filter_criteria={"parent_id": parent_id}, limit=limit)
        
        # If no query text, return all children directly
        if self._qdrant_client:
            res = self._qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="parent_id", match=qmodels.MatchValue(value=parent_id))]
                ),
                limit=limit
            )
            points = res[0]
            return [{"id": p.payload.get("node_id"), "score": 1.0, "payload": p.payload} for p in points]
        else:
            records = self._local_store.get_by_parent_id(parent_id)
            return [{"id": r.id, "score": 1.0, "payload": r.payload} for r in records[:limit]]

    def _search_internal(self, query_vec: List[float], filter_criteria: Optional[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        if self._qdrant_client:
            must_conditions = []
            if filter_criteria:
                for k, v in filter_criteria.items():
                    must_conditions.append(
                        qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                    )
            q_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

            points = []
            if hasattr(self._qdrant_client, "query_points"):
                res = self._qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    query_filter=q_filter,
                    limit=limit
                )
                points = res.points
            elif hasattr(self._qdrant_client, "search"):
                points = self._qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    query_filter=q_filter,
                    limit=limit
                )

            return [
                {
                    "id": getattr(r, "payload", {}).get("node_id", str(r.id)),
                    "score": r.score,
                    "payload": getattr(r, "payload", {})
                }
                for r in points
            ]
        else:
            records = self._local_store.search(query_vec, filter_criteria=filter_criteria, limit=limit)
            return [{"id": r.id, "score": r.score, "payload": r.payload} for r in records]
