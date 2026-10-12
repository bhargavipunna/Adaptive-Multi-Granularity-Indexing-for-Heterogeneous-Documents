"""
Service Container for AMHIE.
Provides centralized lifecycle and state management for document parsing,
feature extraction, adaptive complexity scoring, C++ N-ary tree construction,
HNSW vector indexing, and adaptive RAG generation.
"""

import os
import sys
import uuid
import time
from typing import Dict, Any, Optional, List

# Ensure python root is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from python.parser import DocumentParser, ParsedPage
from python.feature_extractor import FeatureExtractor, PageFeatures
from python.complexity_scorer import ComplexityScorer, ComplexityResult
from python.tree_wrapper import DocumentTree
from python.vector_store import AMHIEVectorStore
from python.retriever import AdaptiveRetriever
from python.rag_pipeline import AMHIERAGPipeline


SAMPLE_HETEROGENEOUS_REPORT = """AMHIE System Architecture & Heterogeneous Document Benchmarks
Page 1 provides a high-level executive overview of the Adaptive Multi-Granularity Hierarchical Indexing Engine.
Traditional RAG systems partition documents uniformly into flat 512-token chunks, leading to catastrophic
layout fragmentation for complex multi-row structures and unnecessary token waste for simple narrative text.
In this chapter, we formulate the mathematical criteria for dynamic ingestion-time indexing.
The architecture computes continuous visual and layout complexity metrics (Visual Dominance Score, Token Density, Layout Dispersion)
to dynamically branch each page into either a coarse parent node or an N-ary fine-grained subtree.

--- PAGE ---
Quarterly Financial Performance & Operational Metrics (FY 2024)
The following consolidated table summarizes revenue, margins, and R&D capital allocation across all operating quarters:

| Quarter | Revenue ($M) | Operating Margin | R&D Investment ($M) | Net Profit ($M) | Cloud Infrastructure ($M) |
|---|---|---|---|---|---|
| Q1 2024 | 124.5 | 22.4% | 34.1 | 18.2 | 12.8 |
| Q2 2024 | 148.9 | 24.8% | 38.6 | 23.5 | 15.2 |
| Q3 2024 | 162.1 | 27.1% | 41.2 | 29.8 | 17.4 |
| Q4 2024 | 195.4 | 31.0% | 45.8 | 42.1 | 19.9 |

Key Performance Highlights: Q4 2024 recorded unprecedented operating margin expansion to 31.0%, propelled by accelerated enterprise AI adoption.
Net profits achieved an all-time record of $42.1M against $45.8M in targeted R&D capital expenditure. Total cloud infrastructure expenditure remained disciplined at $19.9M.

--- PAGE ---
Hardware Acceleration & Vector Indexing Benchmarks
Table 2 details comparative retrieval performance across standard vector chunking and AMHIE adaptive hierarchical indexing:

| Framework | Index Type | Ingestion Latency (ms/p) | Storage Overhead (Vectors/Doc) | Precision@5 | Table Preservation |
|---|---|---|---|---|---|
| Flat 512-Token RAG | Flat Inverted Index | 84.2 | 48.0 | 72.4% | Fragmented (38%) |
| Uniform Sliding Window | HNSW (M=16) | 92.5 | 64.0 | 78.1% | Broken Headers |
| AMHIE (Ours) | N-ary Tree + HNSW | 28.6 | 16.0 | 94.8% | 100% Intact (Markdown) |

Empirical Findings: AMHIE reduces vector index storage footprint by 66.7% while raising Precision@5 to 94.8%. Simple pages remain unified as coarse nodes, preventing redundant chunk indexing.

--- PAGE ---
System Summary and Deployment Guidelines
Section 4 concludes with enterprise deployment specifications for Apple Silicon (macOS) and Linux GPU clusters.
The C++17 Structural Tree Engine (libtree_engine) runs zero-copy logical tree manipulation in sub-millisecond time.
Vector embeddings are projected via BAAI/bge-small-en-v1.5 into a 384-dimensional Euclidean space.
Qdrant HNSW graph index parameters are configured with M=16 and ef_construct=100 for optimal logarithmic recall.
"""


class ServiceContainer:
    """Singleton service container maintaining AMHIE system state."""
    _instance: Optional['ServiceContainer'] = None

    def __init__(self):
        self.parser = DocumentParser()
        self.feature_extractor = FeatureExtractor()
        self.scorer = ComplexityScorer(threshold=0.45)
        self.vector_store = AMHIEVectorStore(collection_name="amhie_production")
        self.retriever = AdaptiveRetriever(self.vector_store)
        self.rag_pipeline = AMHIERAGPipeline(self.retriever)

        # In-memory document registries
        self.active_trees: Dict[str, DocumentTree] = {}
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
        self.document_pages: Dict[str, List[ParsedPage]] = {}
        self.active_doc_id: Optional[str] = None
        self.startup_time = time.time()

    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        if cls._instance is None:
            cls._instance = ServiceContainer()
        return cls._instance

    def ingest_document(self, filename: str, content_bytes: bytes, threshold: float = 0.45) -> Dict[str, Any]:
        """Ingests a document, builds C++ N-ary tree, and indexes into Qdrant."""
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        doc_title = filename or "Uploaded Document"
        self.scorer.threshold = threshold

        # 1. Parse document
        pages: List[ParsedPage] = self.parser.parse_pdf(content_bytes)
        if not pages:
            raise ValueError("No readable pages could be parsed from the document.")

        self.document_pages[doc_id] = pages

        # 2. Construct C++ N-ary Tree
        tree = DocumentTree(doc_id=doc_id, doc_title=doc_title)
        page_summaries = []
        coarse_page_count = 0
        complex_page_count = 0
        total_tables = 0
        total_images = 0
        total_text_blocks = 0

        for page in pages:
            features: PageFeatures = self.feature_extractor.extract_features(page)
            comp_res: ComplexityResult = self.scorer.score_page(features, page.raw_pixmap_base64)
            p_id = f"p{page.page_number}"

            meta = {
                "complexity_score": str(round(comp_res.complexity_score, 4)),
                "vds": str(round(features.visual_dominance_score, 4)),
                "token_density": str(round(features.token_density, 2)),
                "layout_complexity": str(round(features.layout_complexity, 2)),
                "has_children": "true" if comp_res.is_complex else "false",
                "table_count": str(features.table_count),
                "image_count": str(features.image_count)
            }

            if comp_res.is_complex:
                complex_page_count += 1
                page_intro = (
                    f"Page {page.page_number} (Complex Layout). "
                    f"Contains {features.table_count} tables, {features.image_count} visual elements."
                )
                tree.add_page_node(
                    id=p_id,
                    node_type="PAGE_COMPLEX",
                    content=page_intro,
                    page_number=page.page_number,
                    bbox=[0.0, 0.0, page.width, page.height],
                    metadata=meta
                )

                for elem in page.elements:
                    e_meta = {k: str(v) for k, v in elem.metadata.items()}
                    e_meta["parent_id"] = p_id
                    tree.add_child_element(
                        parent_id=p_id,
                        id=elem.element_id,
                        node_type=elem.element_type,
                        content=elem.content,
                        page_number=page.page_number,
                        bbox=elem.bounding_box,
                        metadata=e_meta
                    )
                    if elem.element_type == "TABLE":
                        total_tables += 1
                    elif elem.element_type == "IMAGE":
                        total_images += 1
                    else:
                        total_text_blocks += 1
            else:
                coarse_page_count += 1
                page_summary = page.full_text if page.full_text else f"Page {page.page_number} narrative text."
                tree.add_page_node(
                    id=p_id,
                    node_type="PAGE_COARSE",
                    content=page_summary,
                    page_number=page.page_number,
                    bbox=[0.0, 0.0, page.width, page.height],
                    metadata=meta
                )

            page_summaries.append({
                "page_number": page.page_number,
                "decision": comp_res.decision,
                "complexity_score": round(comp_res.complexity_score, 4),
                "vds": round(features.visual_dominance_score, 4),
                "token_density": round(features.token_density, 2),
                "layout_complexity": round(features.layout_complexity, 2),
                "tables_count": features.table_count,
                "images_count": features.image_count,
                "words_count": features.total_words
            })

        # 3. Index Tree into Qdrant Vector Store (HNSW)
        indexed_vectors = self.vector_store.index_document_tree(tree)
        self.active_trees[doc_id] = tree
        self.active_doc_id = doc_id

        # Estimate savings vs uniform 512-token flat chunking
        est_words = sum(p["words_count"] for p in page_summaries)
        est_flat_chunks = max(1, int(est_words / 350) + (total_tables * 2) + total_images)
        vector_savings_pct = max(0.0, round((1.0 - (indexed_vectors / max(1, est_flat_chunks))) * 100, 1))

        doc_info = {
            "doc_id": doc_id,
            "filename": filename,
            "title": doc_title,
            "total_pages": len(pages),
            "coarse_pages": coarse_page_count,
            "complex_pages": complex_page_count,
            "total_tables_indexed": total_tables,
            "total_images_indexed": total_images,
            "total_text_blocks_indexed": total_text_blocks,
            "total_nodes_in_tree": len(tree),
            "total_vectors_indexed": indexed_vectors,
            "estimated_flat_chunks": est_flat_chunks,
            "storage_reduction_pct": vector_savings_pct,
            "threshold_used": threshold,
            "created_at": time.time(),
            "pages": page_summaries
        }

        self.document_metadata[doc_id] = doc_info
        return doc_info

    def ingest_sample_document(self, threshold: float = 0.45) -> Dict[str, Any]:
        """Loads and indexes the heterogeneous benchmark report for immediate demonstration."""
        sample_bytes = SAMPLE_HETEROGENEOUS_REPORT.encode("utf-8")
        return self.ingest_document("AMHIE_Enterprise_Financial_Benchmark.pdf", sample_bytes, threshold=threshold)

    def get_active_tree(self) -> Optional[DocumentTree]:
        if self.active_doc_id:
            return self.active_trees.get(self.active_doc_id)
        return None

    def query(self, query_text: str) -> Dict[str, Any]:
        """Executes adaptive two-stage retrieval and RAG synthesis."""
        return self.rag_pipeline.generate(query_text).to_dict()


def get_service_container() -> ServiceContainer:
    """Dependency injector for FastAPI routes."""
    return ServiceContainer.get_instance()
