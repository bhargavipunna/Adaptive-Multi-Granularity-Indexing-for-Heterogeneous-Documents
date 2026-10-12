"""
Comprehensive Unit & Integration Test Suite for AMHIE.
Validates:
1. C++17 Structural N-ary Tree Engine (via ctypes & fallback)
2. Feature Extractor (VDS, Td, Lc)
3. Adaptive Complexity Scorer (Tau = 0.45 boundary condition)
4. Vector Store & HNSW Indexing
5. Adaptive Retriever (Two-stage traversal)
6. End-to-End RAG Pipeline & Service
"""

import os
import sys
import unittest

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from python.parser import ParsedPage, ParsedElement, DocumentParser
from python.feature_extractor import FeatureExtractor
from python.complexity_scorer import ComplexityScorer
from python.tree_wrapper import DocumentTree
from python.vector_store import AMHIEVectorStore
from python.retriever import AdaptiveRetriever
from python.rag_pipeline import AMHIERAGPipeline
from api.app import AMHIEService


class TestTreeEngine(unittest.TestCase):
    def test_tree_construction_and_json(self):
        tree = DocumentTree("doc_test", "Test Paper")
        self.assertEqual(len(tree), 1)  # Root node

        # Add coarse page
        res1 = tree.add_page_node("p1", "PAGE_COARSE", "Simple narrative summary", 1, [0, 0, 600, 800], {"vds": "0.05"})
        self.assertTrue(res1)
        self.assertEqual(len(tree), 2)

        # Add complex page
        res2 = tree.add_page_node("p2", "PAGE_COMPLEX", "Complex page with table", 2, [0, 0, 600, 800], {"vds": "0.65"})
        self.assertTrue(res2)
        self.assertEqual(len(tree), 3)

        # Add child table to p2
        res3 = tree.add_child_element("p2", "p2_tab0", "TABLE", "| Col A | Col B |\n|---|---|\n| 1 | 2 |", 2, [50, 100, 550, 350], {"format": "markdown"})
        self.assertTrue(res3)
        self.assertEqual(len(tree), 4)

        # Verify JSON export
        json_str = tree.export_json()
        self.assertIn('"id": "doc_test"', json_str)
        self.assertIn('"id": "p2_tab0"', json_str)
        self.assertIn('"node_type": "TABLE"', json_str)

        # Verify dict representation
        tree_dict = tree.to_dict()
        self.assertEqual(tree_dict["id"], "doc_test")
        self.assertEqual(len(tree_dict["children"]), 2)
        self.assertEqual(tree_dict["children"][1]["children"][0]["id"], "p2_tab0")


class TestFeatureExtractorAndScorer(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()
        self.scorer = ComplexityScorer(threshold=0.45)

    def test_low_complexity_page(self):
        """Page 1: Pure narrative text, no tables, no images -> CS < 0.45"""
        page = ParsedPage(
            page_number=1,
            width=600.0,
            height=800.0,
            full_text="This is an introductory chapter explaining the motivation and background.",
            elements=[
                ParsedElement("p1_t1", "TEXT", "Introductory chapter text.", [50, 50, 550, 200])
            ],
            tables=[],
            images=[],
            text_blocks=[
                ParsedElement("p1_t1", "TEXT", "Introductory chapter text.", [50, 50, 550, 200])
            ]
        )

        feats = self.extractor.extract_features(page)
        self.assertEqual(feats.visual_dominance_score, 0.0)
        self.assertEqual(feats.table_count, 0)
        self.assertEqual(feats.image_count, 0)

        res = self.scorer.score_page(feats)
        self.assertFalse(res.is_complex)
        self.assertEqual(res.decision, "PAGE_COARSE")
        self.assertLess(res.complexity_score, 0.45)

    def test_high_complexity_page(self):
        """Page 2: Dense financial table + technical diagram -> CS >= 0.45"""
        # Table area: (550-50) * (500-150) = 500 * 350 = 175,000
        # Image area: (550-50) * (750-550) = 500 * 200 = 100,000
        # Total Page Area: 600 * 800 = 480,000
        # VDS = (175000 + 100000) / 480000 = 275000 / 480000 = 0.5729
        page = ParsedPage(
            page_number=2,
            width=600.0,
            height=800.0,
            full_text="Results section with financial report table and schematic diagram.",
            elements=[
                ParsedElement("p2_tab1", "TABLE", "| Metric | 2024 |\n|---|---|\n| Revenue | $100M |", [50, 150, 550, 500]),
                ParsedElement("p2_img1", "IMAGE", "[Figure 1: Architecture]", [50, 550, 550, 750])
            ],
            tables=[
                ParsedElement("p2_tab1", "TABLE", "| Metric | 2024 |\n|---|---|\n| Revenue | $100M |", [50, 150, 550, 500])
            ],
            images=[
                ParsedElement("p2_img1", "IMAGE", "[Figure 1: Architecture]", [50, 550, 550, 750])
            ],
            text_blocks=[]
        )

        feats = self.extractor.extract_features(page)
        self.assertGreater(feats.visual_dominance_score, 0.5)
        self.assertEqual(feats.table_count, 1)
        self.assertEqual(feats.image_count, 1)

        res = self.scorer.score_page(feats)
        self.assertTrue(res.is_complex)
        self.assertEqual(res.decision, "PAGE_COMPLEX")
        self.assertGreaterEqual(res.complexity_score, 0.45)


class TestVectorStoreAndAdaptiveRetriever(unittest.TestCase):
    def test_adaptive_two_stage_retrieval(self):
        store = AMHIEVectorStore(collection_name="test_collection")

        # Build sample tree
        tree = DocumentTree("doc_alpha", "Enterprise Cloud Report")
        # Page 1: Coarse (low complexity)
        tree.add_page_node("p1", "PAGE_COARSE", "Cloud infrastructure overview and general guidelines.", 1, [0,0,100,100], {"complexity_score": "0.15"})
        # Page 2: Complex (high complexity)
        tree.add_page_node("p2", "PAGE_COMPLEX", "Financial breakdown and operational statistics.", 2, [0,0,100,100], {"complexity_score": "0.78"})
        # Child elements for Page 2
        tree.add_child_element("p2", "p2_tab0", "TABLE", "| Cloud Provider | Monthly Cost ($) | Uptime |\n| AWS | 45000 | 99.99% |\n| GCP | 38000 | 99.98% |", 2, [10, 10, 90, 40], {"parent_id": "p2"})
        tree.add_child_element("p2", "p2_txt0", "TEXT", "Analysis: GCP proved more cost-effective for deep learning workloads.", 2, [10, 50, 90, 80], {"parent_id": "p2"})

        # Index tree
        indexed = store.index_document_tree(tree)
        self.assertEqual(indexed, 4)

        # Test Adaptive Retriever
        retriever = AdaptiveRetriever(store, top_k_coarse=2, top_k_fine=3)
        results = retriever.retrieve("What are the monthly cloud costs for AWS and GCP?")

        self.assertGreater(len(results), 0)
        # Verify that for Page 2, it traversed into the child TABLE or TEXT block
        retrieved_types = [r.node_type for r in results]
        self.assertTrue("TABLE" in retrieved_types or "TEXT" in retrieved_types or "PAGE_COMPLEX" in retrieved_types)

        # Check traversal path
        paths = [r.traversal_path for r in results]
        self.assertTrue(any("COMPLEX" in p for p in paths))


class TestEndToEndService(unittest.TestCase):
    def test_full_pipeline_ingest_and_query(self):
        service = AMHIEService()

        doc_content = (
            "Executive Overview of AI Infrastructure\n"
            "This report details company progress in Q3.\n"
            "\n--- PAGE ---\n"
            "Performance Benchmarks Table\n"
            "| Model | Latency (ms) | Accuracy |\n"
            "|---|---|---|\n"
            "| AMHIE-Tree | 12.4 | 94.2% |\n"
            "| Baseline | 45.8 | 81.1% |\n"
        ).encode('utf-8')

        stats = service.ingest_document("Benchmark_Report.pdf", doc_content)
        self.assertEqual(stats["total_pages"], 2)
        self.assertEqual(stats["coarse_pages"], 1)
        self.assertEqual(stats["complex_pages"], 1)
        self.assertGreaterEqual(stats["total_tables_indexed"], 1)

        # Query pipeline
        query_res = service.query("What is the latency and accuracy of AMHIE-Tree?")
        self.assertIn("latency_ms", query_res)
        self.assertIn("answer", query_res)
        self.assertGreater(len(query_res["contexts"]), 0)


if __name__ == "__main__":
    unittest.main()
