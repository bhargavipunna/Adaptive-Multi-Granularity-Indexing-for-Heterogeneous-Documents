# AMHIE: Adaptive Multi-Granularity Hierarchical Indexing Engine

[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant_HNSW-red.svg)](https://qdrant.tech/)
[![Embeddings](https://img.shields.io/badge/Embeddings-BGE--small--en--v1.5-purple.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![LLM Generator](https://img.shields.io/badge/Generator-Qwen2.5--0.5B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

**AMHIE** is a high-performance, multimodal Retrieval-Augmented Generation (RAG) indexing framework designed specifically for heterogeneous documents containing mixed narrative text, dense multi-row tables, and visual graphics.

---

## 1. Problem Statement & Novelty

Traditional RAG systems partition documents uniformly using fixed-size flat chunking (e.g., 512 tokens) and raw OCR text flattening:
1. **Computational & Financial Waste:** Simple text-only pages are over-indexed into redundant micro-chunks, multiplying vector database storage and token ingestion costs.
2. **Structural Fragmentation:** Complex tabular data and visual schematics are split across chunk boundaries, permanently destroying row-column alignment and layout context.

### The AMHIE Solution: Ingestion-Time Adaptive Indexing
AMHIE computes continuous visual and layout complexity metrics at ingestion time to dynamically branch each page into a **Structural N-ary Tree**:
- **Low-Complexity Pages ($CS < \tau = 0.45$):** Retained as lightweight **Coarse Parent Nodes** (page-level context) with zero child chunking overhead.
- **High-Complexity Pages ($CS \ge \tau = 0.45$):** Decomposed into an **N-ary Subtree** containing dedicated child nodes for structured Markdown tables, image crops, and text paragraphs.
- **Hierarchical Vector Retrieval:** Top-level coarse nodes are queried first via Qdrant HNSW graph index; the retriever adaptively descends into fine-grained child elements only when a high-complexity page is hit.

---

## 2. System Architecture & Data Flow

```text
[PDF Document Upload]
         │
         ▼
[Stage 1: Preprocessing & Validation (PyMuPDF)]
         │
         ▼
[Stage 2: Multimodal Parsing (MinerU + PyMuPDF)] ──► Extracts Text Blocks, Tables, Images
         │
         ▼
[Stage 3: Feature Extraction Engine (Per Page)]
         ├── Token Density (Td)
         ├── Visual Dominance Score (VDS) = (Table Area + Image Area) / Total Page Area
         └── Table Count, Image Count, Layout Complexity (Lc)
         │
         ▼
[Stage 4: Adaptive Complexity Scoring (Qwen2-VL)] ──► Complexity Score (CS)
         │
         ├── IF CS < Threshold (Tau=0.45): Low Complexity  ──► Coarse Node (Page-Level Text)
         └── IF CS >= Threshold (Tau=0.45): High Complexity ──► Fine-Grained Split (N-ary Subtree)
         │
         ▼
[Stage 5: Structural N-ary Tree Construction Engine (C++17)]
         └── Constructs logical hierarchical document layout (libtree_engine.dylib/.so)
         │
         ▼
[Stage 6: Vector Embedding & Storage (Qdrant + HNSW)]
         ├── Embeddings generated via BAAI/bge-small-en-v1.5
         └── Ingested into Qdrant Vector Store with HNSW Graph Index (m=16, ef=100)
         │
         ▼
[Stage 7: Adaptive Retrieval & Generation]
         ├── User Query ──► Adaptive HNSW Tree Traversal Search
         └── Context Selection & Re-ranking ──► Qwen2.5-0.5B-Instruct ──► Final Answer
```

---

## 3. Mathematical Formulations

### Visual Dominance Score ($VDS$)
The ratio of area consumed by structured elements (tables and figures) to the total page area:
$$\text{VDS} = \frac{\sum \text{Area}(\text{Table}_i) + \sum \text{Area}(\text{Image}_j)}{\text{Page Width} \times \text{Page Height}}$$

### Scaled Token Density ($T_d$)
Approximates token compactness per unit area:
$$T_d = \left(\frac{N_{\text{words}} \times 1.3}{\text{Page Area}}\right) \times 10^4$$

### Layout Complexity ($L_c$)
Captures structural dispersion, multi-column presence, and element boundaries:
$$L_c = (N_{\text{text}} \times 0.4) + (N_{\text{tables}} \times 3.5) + (N_{\text{images}} \times 1.5) + (N_{\text{cols}} - 1) \times 3.0$$

### Continuous Complexity Score ($CS$)
$$CS = w_1 \cdot \text{VDS} + w_2 \cdot \min\left(1.0, N_{\text{tables}} \cdot 0.8 + N_{\text{images}} \cdot 0.45\right) + w_3 \cdot \min\left(1.0, \frac{L_c}{10.0}\right) + w_4 \cdot \min\left(1.0, \frac{T_d}{60.0}\right) + \mathbf{1}_{\{N_{\text{tables}} > 0\}} \cdot 0.15$$
Where $w_1 = 0.35, w_2 = 0.35, w_3 = 0.20, w_4 = 0.10$.

- If $CS < \tau = 0.45 \implies$ **Low Complexity** (`PAGE_COARSE`)
- If $CS \ge \tau = 0.45 \implies$ **High Complexity** (`PAGE_COMPLEX`)

---

## 4. Vector Search vs B-Trees

| Dimension | B-Trees / B+ Trees | HNSW (Hierarchical Navigable Small World) |
| :--- | :--- | :--- |
| **Search Space** | 1-Dimensional scalar keys (e.g. integers, strings) | High-dimensional continuous metric vectors ($\mathbb{R}^d$) |
| **Curse of Dimensionality** | Fails completely for $d > 10$; degrades to $O(N)$ scan | Efficient logarithmic search $O(\log N)$ in $\mathbb{R}^{384}$ |
| **Similarity Metric** | Exact range and equality match | Cosine similarity, Euclidean distance, Dot product |
| **AMHIE Role** | C++ N-ary Tree handles **logical document hierarchy** | Qdrant HNSW handles **semantic similarity search** |

---

## 5. Project Directory Layout

```
amhie_project/
├── csrc/
│   ├── tree_engine.cpp       # C++17 N-ary Tree implementation & C-ABI exports
│   ├── tree_engine.h         # Header file for C++ Tree
│   ├── libtree_engine.dylib  # Compiled shared library (macOS)
│   └── libtree_engine.so     # Compiled shared library (Linux)
├── python/
│   ├── __init__.py           # Package interface
│   ├── parser.py             # PyMuPDF + MinerU Integration & table extraction
│   ├── feature_extractor.py  # Calculates VDS, Token Density, Layout Score
│   ├── complexity_scorer.py  # Qwen2-VL & Layout Complexity Scoring Engine
│   ├── tree_wrapper.py       # C++ Tree Python Binding Wrapper (ctypes / pybind11)
│   ├── vector_store.py       # Qdrant Vector Store & BGE-small Embeddings (HNSW)
│   ├── retriever.py          # Adaptive HNSW Tree Traversal Search
│   └── rag_pipeline.py       # Qwen2.5-0.5B Generation Pipeline
├── api/
│   └── app.py                # FastAPI REST Endpoints (/upload_pdf, /search_query)
├── ui/
│   └── app.py                # Streamlit Dashboard UI
├── tests/
│   └── test_amhie.py         # Unit and integration test suite
├── Makefile                  # C++ build automation
├── requirements.txt          # Python dependencies
└── README.md                 # Full documentation
```

---

## 6. Quickstart & Installation

### Step 1: Clone and Setup Virtual Environment
```bash
cd amhie_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Compile the C++17 Structural Tree Engine
```bash
make
# or directly with clang++:
clang++ -std=c++17 -O3 -fPIC -shared csrc/tree_engine.cpp -o csrc/libtree_engine.dylib
cp csrc/libtree_engine.dylib csrc/libtree_engine.so
```

### Step 3: Run the Test Suite
```bash
python3 tests/test_amhie.py
```

### Step 4: Launch the FastAPI REST Server
```bash
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5: Launch the Streamlit Dashboard
```bash
streamlit run ui/app.py
```

---

## 7. API Endpoints

### `POST /upload_pdf`
Uploads and parses a PDF, extracts layout features, builds the C++ N-ary tree, and indexes nodes into Qdrant.
```bash
curl -X POST "http://localhost:8000/upload_pdf" \
  -F "file=@sample_report.pdf"
```
**Response:**
```json
{
  "doc_id": "doc_a1b2c3d4",
  "filename": "sample_report.pdf",
  "total_pages": 4,
  "coarse_pages": 2,
  "complex_pages": 2,
  "total_tables_indexed": 3,
  "total_images_indexed": 1,
  "total_nodes_in_tree": 12,
  "total_vectors_indexed": 11,
  "tree_json": { ... }
}
```

### `POST /search_query`
Performs two-stage adaptive retrieval and generates synthesized answer via Qwen2.5-0.5B-Instruct.
```bash
curl -X POST "http://localhost:8000/search_query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the Q4 2024 operating margins and revenue?", "top_k": 3}'
```

### `GET /tree_structure/{doc_id}`
Returns the serialized hierarchical N-ary tree for frontend rendering.

---

## 8. License & Attribution
Designed and built for High-Performance Enterprise Multimodal Document Retrieval.
