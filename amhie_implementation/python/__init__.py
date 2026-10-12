"""
AMHIE: Adaptive Multi-Granularity Hierarchical Indexing Engine
"""

from .parser import DocumentParser, ParsedPage, ParsedElement
from .feature_extractor import FeatureExtractor, PageFeatures
from .complexity_scorer import ComplexityScorer, ComplexityResult
from .tree_wrapper import DocumentTree, TreeNode
from .vector_store import AMHIEVectorStore
from .retriever import AdaptiveRetriever, RetrievedContext
from .rag_pipeline import AMHIERAGPipeline

__version__ = "1.0.0"
__all__ = [
    "DocumentParser",
    "ParsedPage",
    "ParsedElement",
    "FeatureExtractor",
    "PageFeatures",
    "ComplexityScorer",
    "ComplexityResult",
    "DocumentTree",
    "TreeNode",
    "AMHIEVectorStore",
    "AdaptiveRetriever",
    "RetrievedContext",
    "AMHIERAGPipeline",
]
