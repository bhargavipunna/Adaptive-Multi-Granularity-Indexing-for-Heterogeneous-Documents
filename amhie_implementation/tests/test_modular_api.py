"""
Test Suite for AMHIE Modular Storage & API Routers.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.app import app
from api.services.service_container import get_service_container


class TestModularAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Ensure sample document is loaded
        container = get_service_container()
        container.ingest_sample_document(threshold=0.45)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["engine"], "AMHIE")

    def test_storage_overview(self):
        resp = self.client.get("/api/storage/overview")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("storage_tiers", data)
        self.assertIn("tier_1_multimodal", data["storage_tiers"])
        self.assertIn("tier_2_cpp_tree", data["storage_tiers"])
        self.assertIn("tier_3_qdrant_hnsw", data["storage_tiers"])
        self.assertIn("tier_4_complexity_store", data["storage_tiers"])
        self.assertIn("storage_economy", data)

    def test_storage_vectors(self):
        resp = self.client.get("/api/storage/vectors")
        self.assertEqual(resp.status_code, 200)
        vectors = resp.json()
        self.assertIsInstance(vectors, list)
        self.assertGreater(len(vectors), 0)
        # Check first vector point
        p0 = vectors[0]
        self.assertIn("node_id", p0)
        self.assertIn("node_type", p0)
        self.assertIn("vector_dimensions", p0)
        self.assertEqual(p0["vector_dimensions"], 384)

    def test_storage_hnsw(self):
        resp = self.client.get("/api/storage/hnsw")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["parameters"]["m"], 16)
        self.assertEqual(data["parameters"]["ef_construct"], 100)
        self.assertEqual(data["metric"], "Cosine Similarity")

    def test_tree_hierarchy(self):
        container = get_service_container()
        doc_id = container.active_doc_id
        self.assertIsNotNone(doc_id)

        resp = self.client.get(f"/api/tree/{doc_id}")
        self.assertEqual(resp.status_code, 200)
        tree = resp.json()
        self.assertEqual(tree["id"], doc_id)
        self.assertIn("children", tree)

    def test_frontend_serving(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("AMHIE", resp.text)
        self.assertIn("Adaptive Multi-Granularity Hierarchical Indexing Engine", resp.text)

    def test_rag_query(self):
        resp = self.client.post("/api/rag/generate", json={
            "query": "What were the Q4 2024 operating margins and revenue?",
            "top_k": 3
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        self.assertIn("contexts", data)
        self.assertIn("retrieval_paths", data)
        self.assertGreater(len(data["contexts"]), 0)


if __name__ == "__main__":
    unittest.main()
