"""
AMHIE Streamlit Dashboard
(Adaptive Multi-Granularity Hierarchical Indexing Engine)
6-Tab Professional Demo Interface showing the complete backend pipeline.
"""

import os, sys, json, time, base64
from typing import Dict, Any, List, Optional

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from python.parser import DocumentParser, ParsedPage
from python.feature_extractor import FeatureExtractor, PageFeatures
from python.complexity_scorer import ComplexityScorer, ComplexityResult
from python.tree_wrapper import DocumentTree
from python.vector_store import AMHIEVectorStore
from python.retriever import AdaptiveRetriever
from python.rag_pipeline import AMHIERAGPipeline
import streamlit as st


# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMHIE: Adaptive Multi-Granularity Hierarchical Indexing Engine",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Sample Document ─────────────────────────────────────────────────────────
SAMPLE_DOC = """AMHIE System Architecture & Heterogeneous Document Benchmarks
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
Net profits achieved an all-time record of $42.1M against $45.8M in targeted R&D capital expenditure.

--- PAGE ---
Hardware Acceleration & Vector Indexing Benchmarks
Table 2 details comparative retrieval performance across standard vector chunking and AMHIE adaptive hierarchical indexing:

| Framework | Index Type | Ingestion Latency (ms/p) | Storage Overhead (Vectors/Doc) | Precision@5 | Table Preservation |
|---|---|---|---|---|---|
| Flat 512-Token RAG | Flat Inverted Index | 84.2 | 48.0 | 72.4% | Fragmented (38%) |
| Uniform Sliding Window | HNSW (M=16) | 92.5 | 64.0 | 78.1% | Broken Headers |
| AMHIE (Ours) | N-ary Tree + HNSW | 28.6 | 16.0 | 94.8% | 100% Intact (Markdown) |

Empirical Findings: AMHIE reduces vector index storage footprint by 66.7% while raising Precision@5 to 94.8%.
Simple pages remain unified as coarse nodes, preventing redundant chunk indexing.

--- PAGE ---
System Summary and Deployment Guidelines
Section 4 concludes with enterprise deployment specifications for Apple Silicon (macOS) and Linux GPU clusters.
The C++17 Structural Tree Engine (libtree_engine) runs zero-copy logical tree manipulation in sub-millisecond time.
Vector embeddings are projected via BAAI/bge-small-en-v1.5 into a 384-dimensional Euclidean space.
Qdrant HNSW graph index parameters are configured with M=16 and ef_construct=100 for optimal logarithmic recall.
"""


# ─── Session State Initialization ────────────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.parser = DocumentParser()
    st.session_state.feature_extractor = FeatureExtractor()
    st.session_state.scorer = ComplexityScorer(threshold=0.45)
    st.session_state.vector_store = AMHIEVectorStore(collection_name="amhie_streamlit")
    st.session_state.retriever = AdaptiveRetriever(st.session_state.vector_store)
    st.session_state.rag = AMHIERAGPipeline(st.session_state.retriever)
    st.session_state.current_tree = None
    st.session_state.ingest_stats = None
    st.session_state.processing_log = []
    st.session_state.pages_data = []
    st.session_state.page_features = []
    st.session_state.doc_filename = None
    st.session_state.initialized = True


def log_step(msg: str, duration_ms: float = 0):
    """Appends a timestamped log entry to the processing log."""
    ts = time.strftime("%H:%M:%S")
    dur = f" [{duration_ms:.1f} ms]" if duration_ms > 0 else ""
    st.session_state.processing_log.append(f"[{ts}] {msg}{dur}")


def process_document(raw_input, filename: str):
    """
    Full AMHIE pipeline: Parse → Extract Features → Score Complexity →
    Build Tree → Index Vectors. All timing and details are logged.
    """
    # Reset state
    st.session_state.processing_log = []
    st.session_state.pages_data = []
    st.session_state.page_features = []
    st.session_state.current_tree = None
    st.session_state.ingest_stats = None
    st.session_state.doc_filename = filename

    total_start = time.time()
    log_step(f"═══ Starting AMHIE Pipeline for: {filename} ═══")

    try:
        # ── Step 1: Document Parsing ──────────────────────────────────────
        t0 = time.time()
        log_step("Step 1/5: Document Parsing (PyMuPDF / Text Fallback)")

        if isinstance(raw_input, str):
            content_bytes = raw_input.encode("utf-8")
        elif isinstance(raw_input, bytes):
            content_bytes = raw_input
        else:
            # Streamlit UploadedFile
            content_bytes = raw_input.read()

        pages: List[ParsedPage] = st.session_state.parser.parse_pdf(content_bytes)
        if not pages:
            st.error("No readable pages could be parsed from the document.")
            log_step("ERROR: No pages parsed.")
            return

        parse_ms = (time.time() - t0) * 1000
        log_step(f"  → Parsed {len(pages)} pages successfully", parse_ms)
        for p in pages:
            log_step(f"    Page {p.page_number}: {len(p.tables)} tables, {len(p.images)} images, {len(p.text_blocks)} text blocks, {len(p.full_text.split())} words")

        # ── Step 2 & 3: Feature Extraction + Complexity Scoring ───────────
        t1 = time.time()
        log_step("Step 2/5: Feature Extraction (VDS, Td, Lc)")
        log_step("Step 3/5: Complexity Scoring (CS vs τ)")

        tau = st.session_state.get("tau_val", 0.45)
        st.session_state.scorer.threshold = tau
        doc_id = f"doc_{int(time.time())}"

        tree = DocumentTree(doc_id=doc_id, doc_title=filename)
        total_coarse = 0
        total_complex = 0
        total_tables = 0
        total_images = 0
        total_text_blocks = 0

        for page in pages:
            tp = time.time()
            features: PageFeatures = st.session_state.feature_extractor.extract_features(page)
            comp: ComplexityResult = st.session_state.scorer.score_page(features, page.raw_pixmap_base64)
            page_ms = (time.time() - tp) * 1000

            p_id = f"p{page.page_number}"
            meta = {
                "complexity_score": str(round(comp.complexity_score, 4)),
                "vds": str(round(features.visual_dominance_score, 4)),
                "token_density": str(round(features.token_density, 2)),
                "layout_complexity": str(round(features.layout_complexity, 2)),
                "has_children": "true" if comp.is_complex else "false",
                "table_count": str(features.table_count),
                "image_count": str(features.image_count),
            }

            # ── Step 4: Tree Construction ─────────────────────────────────
            if comp.is_complex:
                total_complex += 1
                page_intro = (
                    f"Page {page.page_number} (Complex Layout). "
                    f"Contains {features.table_count} tables, {features.image_count} visual elements."
                )
                tree.add_page_node(
                    id=p_id, node_type="PAGE_COMPLEX",
                    content=page_intro, page_number=page.page_number,
                    bbox=[0.0, 0.0, page.width, page.height], metadata=meta
                )
                for elem in page.elements:
                    e_meta = {k: str(v) for k, v in elem.metadata.items()}
                    e_meta["parent_id"] = p_id
                    tree.add_child_element(
                        parent_id=p_id, id=elem.element_id,
                        node_type=elem.element_type, content=elem.content,
                        page_number=page.page_number, bbox=elem.bounding_box,
                        metadata=e_meta
                    )
                    if elem.element_type == "TABLE":
                        total_tables += 1
                    elif elem.element_type == "IMAGE":
                        total_images += 1
                    else:
                        total_text_blocks += 1
            else:
                total_coarse += 1
                tree.add_page_node(
                    id=p_id, node_type="PAGE_COARSE",
                    content=page.full_text, page_number=page.page_number,
                    bbox=[0.0, 0.0, page.width, page.height], metadata=meta
                )

            # Store per-page data for display
            st.session_state.pages_data.append({
                "Page": page.page_number,
                "Decision": comp.decision,
                "CS": round(comp.complexity_score, 4),
                "VDS": round(features.visual_dominance_score, 4),
                "Td": round(features.token_density, 2),
                "Lc": round(features.layout_complexity, 2),
                "Tables": features.table_count,
                "Images": features.image_count,
                "Words": features.total_words,
            })
            st.session_state.page_features.append(features.to_dict())

            log_step(
                f"  → Page {page.page_number}: CS={comp.complexity_score:.4f} "
                f"{'≥' if comp.is_complex else '<'} τ={tau} → {comp.decision}",
                page_ms
            )

        feat_score_ms = (time.time() - t1) * 1000
        log_step(f"Step 4/5: Tree Construction Complete ({len(tree)} nodes)", feat_score_ms)

        # ── Step 5: Vector Indexing ───────────────────────────────────────
        t2 = time.time()
        log_step("Step 5/5: Vector Indexing (HNSW Upsert)")
        st.session_state.current_tree = tree
        indexed_count = st.session_state.vector_store.index_document_tree(tree)
        index_ms = (time.time() - t2) * 1000
        log_step(f"  → Indexed {indexed_count} vectors into Qdrant HNSW", index_ms)

        # ── Compute Storage Savings ───────────────────────────────────────
        total_words = sum(p["Words"] for p in st.session_state.pages_data)
        est_flat = max(1, int(total_words / 350) + (total_tables * 2) + total_images)
        savings_pct = max(0.0, round((1.0 - indexed_count / max(1, est_flat)) * 100, 1))

        total_ms = (time.time() - total_start) * 1000
        log_step(f"═══ Pipeline Complete: {indexed_count} vectors, {len(tree)} nodes ═══", total_ms)

        st.session_state.ingest_stats = {
            "total_pages": len(pages),
            "coarse_pages": total_coarse,
            "complex_pages": total_complex,
            "tree_nodes": len(tree),
            "indexed_vectors": indexed_count,
            "total_tables": total_tables,
            "total_images": total_images,
            "total_text_blocks": total_text_blocks,
            "estimated_flat_chunks": est_flat,
            "storage_savings_pct": savings_pct,
            "total_latency_ms": round(total_ms, 1),
            "threshold_used": tau,
        }

    except Exception as e:
        log_step(f"ERROR: {e}")
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/tree-structure.png", width=64)
    st.header("📄 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload PDF / TXT / MD", type=["pdf", "txt", "md"],
        help="Upload any document. PDFs are parsed with PyMuPDF; text files use the fallback parser."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Process Upload", disabled=(uploaded_file is None), use_container_width=True):
            process_document(uploaded_file, uploaded_file.name)
    with col2:
        if st.button("📋 Load Sample", use_container_width=True):
            process_document(SAMPLE_DOC, "AMHIE_Benchmark_Report.txt")

    st.markdown("---")
    st.subheader("⚙️ Configuration")
    tau_val = st.slider(
        "Complexity Threshold (τ)", min_value=0.1, max_value=0.9,
        value=0.45, step=0.05, key="tau_val",
        help="Pages with CS ≥ τ are decomposed into fine-grained subtrees"
    )

    st.markdown("---")
    st.subheader("🔧 Engine Stack")
    st.markdown("""
    | Component | Technology |
    |---|---|
    | **Parser** | PyMuPDF (fitz) |
    | **Tree Engine** | C++17 `libtree_engine.dylib` |
    | **Embeddings** | `BAAI/bge-small-en-v1.5` (384d) |
    | **Vector DB** | Qdrant HNSW (M=16, ef=100) |
    | **Distance** | Cosine Similarity |
    | **Generator** | Extractive Reasoner |
    """)


# ─── Main Content ────────────────────────────────────────────────────────────
st.title("🌳 AMHIE: Adaptive Multi-Granularity Hierarchical Indexing Engine")
st.caption("Ingestion-Time Adaptive Indexing · C++17 Structural N-ary Tree · Qdrant HNSW Graph Index")

if st.session_state.current_tree is None:
    st.info("👋 **Welcome to AMHIE!** Upload a document or click **Load Sample** in the sidebar to begin the demo.")
    st.markdown("""
    ### How AMHIE Works
    1. **Parse** — Extract text blocks, tables (Markdown), and images from PDF pages
    2. **Score** — Compute per-page complexity: VDS, Td, Lc → CS formula
    3. **Branch** — CS ≥ τ → decompose into N-ary subtree; CS < τ → keep as coarse node
    4. **Index** — Embed nodes with BGE encoder → store in Qdrant HNSW vector graph
    5. **Retrieve** — Adaptive 2-stage search: coarse routing → fine-grained drill-down
    6. **Answer** — Extractive synthesis with table-aware context formatting
    """)
    st.stop()

# ─── 6-Tab Dashboard ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Pipeline Overview & Metrics",
    "🌲 Document Tree Visualization",
    "⚡ Vector Store & HNSW Index",
    "🔍 Adaptive RAG Q&A",
    "🔬 Backend Processing Log",
    "📜 Raw Data Inspector",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Pipeline Overview & Metrics
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    stats = st.session_state.ingest_stats

    st.subheader(f"📄 Document: {st.session_state.doc_filename}")

    # Metric Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Pages", stats["total_pages"])
    m2.metric("🟢 Coarse Pages", stats["coarse_pages"])
    m3.metric("🟠 Complex Pages", stats["complex_pages"])
    m4.metric("🌲 Tree Nodes", stats["tree_nodes"])
    m5.metric("⚡ Indexed Vectors", stats["indexed_vectors"])
    m6.metric("💾 Storage Savings", f"{stats['storage_savings_pct']}%")

    st.markdown("---")

    # CS Formula
    st.subheader("Complexity Scoring Formula")
    st.latex(r"CS = \underbrace{0.35 \cdot VDS}_{\text{Visual Dominance}} + \underbrace{0.35 \cdot \text{Elems}}_{\text{Table+Image}} + \underbrace{0.20 \cdot L_c}_{\text{Layout}} + \underbrace{0.10 \cdot T_d}_{\text{Token Density}} + \underbrace{0.15}_{\text{Table Bonus}}")
    st.caption(f"Decision boundary: τ = {stats['threshold_used']} · CS ≥ τ → PAGE_COMPLEX (N-ary subtree) · CS < τ → PAGE_COARSE (single node)")

    st.markdown("---")

    # Per-Page Analysis Table
    st.subheader("📑 Per-Page Complexity Analysis")
    display_data = []
    for p in st.session_state.pages_data:
        badge = "🟢 COARSE" if p["Decision"] == "PAGE_COARSE" else "🟠 COMPLEX"
        display_data.append({
            "Page": p["Page"],
            "Decision": badge,
            "CS": p["CS"],
            "VDS": p["VDS"],
            "Td": p["Td"],
            "Lc": p["Lc"],
            "Tables": p["Tables"],
            "Images": p["Images"],
            "Words": p["Words"],
        })
    st.dataframe(display_data, use_container_width=True)

    # Storage comparison
    st.subheader("💾 Storage Efficiency Comparison")
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("AMHIE Vectors", stats["indexed_vectors"])
    scol2.metric("Est. Flat 512-Token Chunks", stats["estimated_flat_chunks"])
    scol3.metric("Vector Reduction", f"{stats['storage_savings_pct']}%",
                 delta=f"-{stats['estimated_flat_chunks'] - stats['indexed_vectors']} vectors",
                 delta_color="inverse")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Document Tree Visualization
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌲 N-ary Document Tree Structure")
    st.caption("Color legend: 📚 Document Root · 🟢 Coarse Page · 🟠 Complex Page · 📊 Table · 🖼️ Image · 📝 Text Block")

    tree_dict = st.session_state.current_tree.to_dict()

    # ── Interactive Graphviz Tree Diagram ────────────────────────────────────
    def build_graphviz_dot(root_node: Dict[str, Any], direction: str = "TB") -> str:
        dot_lines = [
            "digraph AMHIE_Document_Tree {",
            f"  graph [rankdir={direction}, bgcolor=transparent, nodesep=0.35, ranksep=0.55];",
            '  node [fontname="Helvetica,Arial,sans-serif", fontsize=10, style="filled,rounded", shape=box, penwidth=1.5];',
            '  edge [color="#78909C", penwidth=1.3, arrowhead=vee, arrowsize=0.8];'
        ]

        def _escape_lbl(txt: str) -> str:
            return txt.replace('"', '\\"').replace('\n', '\\n')

        def _traverse(n: Dict[str, Any], parent_id: Optional[str] = None):
            nid = n.get("id", "root")
            ntype = n.get("node_type", "UNKNOWN")
            dot_id = "node_" + "".join(c if c.isalnum() else "_" for c in nid)
            meta = n.get("metadata", {})

            if ntype == "DOCUMENT":
                title_preview = _escape_lbl(n.get("content", "Document")[:24])
                lbl = f"📚 DOCUMENT ROOT\\n{title_preview}"
                dot_lines.append(
                    f'  {dot_id} [label="{lbl}", fillcolor="#E8EAF6", color="#3F51B5", fontcolor="#1A237E", shape=hexagon, fontsize=11];'
                )
            elif ntype == "PAGE_COARSE":
                cs = meta.get("complexity_score", "")
                cs_txt = f"\\nCS = {cs}" if cs else ""
                lbl = f"🟢 Page {n.get('page_number', '?')}\\n[PAGE_COARSE]{cs_txt}\\n(Leaf: Single Chunk)"
                dot_lines.append(
                    f'  {dot_id} [label="{lbl}", fillcolor="#E8F5E9", color="#2E7D32", fontcolor="#1B5E20"];'
                )
            elif ntype == "PAGE_COMPLEX":
                cs = meta.get("complexity_score", "")
                cs_txt = f"\\nCS = {cs}" if cs else ""
                lbl = f"🟠 Page {n.get('page_number', '?')}\\n[PAGE_COMPLEX]{cs_txt}\\n(N-ary Subtree)"
                dot_lines.append(
                    f'  {dot_id} [label="{lbl}", fillcolor="#FFF3E0", color="#E65100", fontcolor="#BF360C"];'
                )
            elif ntype == "TABLE":
                rows = meta.get("rows", "")
                cols = meta.get("cols", "")
                dim_txt = f" ({rows}x{cols})" if rows and cols else ""
                lbl = f"📊 TABLE{dim_txt}\\n{_escape_lbl(nid)}"
                dot_lines.append(
                    f'  {dot_id} [label="{lbl}", fillcolor="#E1F5FE", color="#0288D1", fontcolor="#01579B"];'
                )
            elif ntype == "IMAGE":
                lbl = f"🖼️ IMAGE\\n{_escape_lbl(nid)}"
                dot_lines.append(
                    f'  {dot_id} [label="{lbl}", fillcolor="#F3E5F5", color="#8E24AA", fontcolor="#4A148C"];'
                )
            else:
                lbl = f"📝 TEXT BLOCK\\n{_escape_lbl(nid)}"
                dot_lines.append(
                    f'  {dot_id} [label="{lbl}", fillcolor="#FAFAFA", color="#757575", fontcolor="#212121"];'
                )

            if parent_id:
                dot_lines.append(f"  {parent_id} -> {dot_id};")

            for child in n.get("children", []):
                _traverse(child, dot_id)

        _traverse(root_node)
        dot_lines.append("}")
        return "\n".join(dot_lines)

    c_tree_top1, c_tree_top2 = st.columns([3, 1])
    with c_tree_top1:
        st.markdown("#### 🌿 Live Structural Tree Graph")
    with c_tree_top2:
        tree_orient = st.selectbox("Orientation", ["Top-to-Bottom", "Left-to-Right"], index=0, key="tree_orient_sel")

    rankdir = "TB" if tree_orient == "Top-to-Bottom" else "LR"
    dot_source = build_graphviz_dot(tree_dict, direction=rankdir)
    st.graphviz_chart(dot_source, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Detailed Node Explorer")

    def render_tree(node: Dict, level: int = 0):
        ntype = node.get("node_type", "UNKNOWN")
        nid = node.get("id", "?")
        page = node.get("page_number", "")
        meta = node.get("metadata", {})
        children = node.get("children", [])
        content = node.get("content", "")

        # Emoji mapping
        icons = {
            "DOCUMENT": "📚", "PAGE_COARSE": "🟢", "PAGE_COMPLEX": "🟠",
            "TABLE": "📊", "IMAGE": "🖼️", "TEXT": "📝"
        }
        icon = icons.get(ntype, "📄")
        cs_str = f" · CS={meta.get('complexity_score', 'N/A')}" if "complexity_score" in meta else ""

        label = f"{icon} **{nid}** ({ntype}){cs_str}"
        if page:
            label += f" — Page {page}"

        with st.expander(label, expanded=(level < 2)):
            if content:
                preview = content[:400] + ("..." if len(content) > 400 else "")
                st.text(preview)
            if meta:
                st.caption(f"Metadata: {json.dumps(meta, indent=2)}")
            if node.get("bounding_box"):
                st.caption(f"Bounding Box: {node['bounding_box']}")

        for child in children:
            render_tree(child, level + 1)

    render_tree(tree_dict)

    # BFS Node Listing
    st.markdown("---")
    st.subheader("📋 BFS Node Listing")
    all_nodes = st.session_state.current_tree.get_all_nodes()
    node_table = []
    for n in all_nodes:
        node_table.append({
            "ID": n.get("id"),
            "Type": n.get("node_type"),
            "Page": n.get("page_number", ""),
            "Parent": n.get("parent_id", "—"),
            "Children": "Yes" if n.get("metadata", {}).get("has_children") == "true" else "No",
            "Content (first 60 chars)": (n.get("content", "")[:60] + "...") if len(n.get("content", "")) > 60 else n.get("content", ""),
        })
    st.dataframe(node_table, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Vector Store & HNSW Index
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚡ Qdrant HNSW Vector Index")

    # HNSW Config
    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    hcol1.metric("Vector Dimensions", 384)
    hcol2.metric("HNSW M", 16)
    hcol3.metric("ef_construct", 100)
    hcol4.metric("Distance Metric", "Cosine")

    st.markdown("---")

    # Pipeline visualization
    st.subheader("Vector Indexing Pipeline")
    st.markdown("""
    ```
    Document Page ──→ Text / Table Content
                          │
                          ▼
              ┌─────────────────────────┐
              │  BAAI/bge-small-en-v1.5 │
              │   Sentence Transformer  │
              │     (384-dim output)    │
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Qdrant HNSW Graph     │
              │  M=16, ef_construct=100 │
              │   Cosine Distance       │
              └─────────────────────────┘
    ```
    """)

    st.markdown("---")

    # Indexed Points Table
    st.subheader("📦 All Indexed Vector Points")
    nodes = st.session_state.current_tree.get_all_nodes()
    vec_table = []
    for n in nodes:
        if n.get("node_type") == "DOCUMENT":
            continue
        content = n.get("content", "")
        vec_table.append({
            "Node ID": n.get("id"),
            "Type": n.get("node_type"),
            "Page": n.get("page_number"),
            "Parent": n.get("parent_id", "root"),
            "CS": n.get("metadata", {}).get("complexity_score", "N/A"),
            "Content Preview": (content[:80] + "...") if len(content) > 80 else content,
        })
    st.dataframe(vec_table, use_container_width=True)

    # Sample Embedding
    st.markdown("---")
    st.subheader("🔢 Sample Embedding Vector")
    st.caption("Showing first 20 of 384 dimensions for a sample text:")
    sample_text = "Q4 2024 operating margin expansion"
    sample_vec = st.session_state.vector_store.embed_text(sample_text)
    st.code(f"Text: \"{sample_text}\"\nDimensions: {len(sample_vec)}\nFirst 20 values: {[round(v, 6) for v in sample_vec[:20]]}", language="python")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Adaptive RAG Q&A
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Adaptive RAG Search & Generation")

    # Explain the 2-stage process
    st.markdown("""
    **How Adaptive Retrieval Works:**
    1. **Stage 1 — Coarse Routing:** Query the HNSW index for top page-level nodes
    2. **Stage 2 — Fine-Grained Drill-Down:** For `PAGE_COMPLEX` matches, descend into child elements (tables, images, text blocks) and re-rank by similarity
    3. **Generation:** Format retrieved multi-granularity contexts → Extractive synthesis with citations
    """)

    st.markdown("---")

    query = st.text_input(
        "Ask a question about the document:",
        value="What was the operating margin and net profit in Q4 2024?",
        placeholder="Type your question here..."
    )

    sample_queries = [
        "What was the operating margin and net profit in Q4 2024?",
        "How does AMHIE compare to flat 512-token RAG in Precision@5?",
        "What is the total R&D investment across all quarters?",
        "What HNSW parameters does AMHIE use?",
    ]

    st.caption("Sample queries: " + " · ".join([f"`{q[:40]}...`" for q in sample_queries]))

    if st.button("🚀 Run Adaptive RAG", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Executing adaptive HNSW traversal..."):
                response = st.session_state.rag.generate(query)

            # Answer
            st.markdown("### 💡 Generated Answer")
            st.success(response.answer)

            # Metrics row
            lcol1, lcol2 = st.columns(2)
            lcol1.metric("Latency", f"{response.latency_ms:.1f} ms")
            lcol2.metric("Generator", response.model_used)

            # Traversal Paths
            st.markdown("### 🧭 Adaptive Traversal Paths")
            for i, path in enumerate(response.retrieval_paths):
                st.code(f"Path {i+1}: {path}", language="text")

            # Retrieved Contexts
            st.markdown("### 📦 Retrieved Context Nodes")
            for i, ctx in enumerate(response.contexts):
                with st.expander(
                    f"[{ctx.node_type}] {ctx.node_id} — Page {ctx.page_number} "
                    f"(Score: {ctx.similarity_score:.4f})"
                ):
                    st.markdown(f"**Traversal:** `{ctx.traversal_path}`")
                    st.markdown(f"**Bounding Box:** `{ctx.bounding_box}`")
                    st.markdown(f"**Parent:** `{ctx.parent_id}`")
                    st.markdown("**Content:**")
                    st.markdown(ctx.content)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Backend Processing Log
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔬 Backend Processing Pipeline Log")
    st.caption("Step-by-step execution trace of the AMHIE ingestion pipeline")

    if st.session_state.processing_log:
        # Pipeline stages diagram
        st.markdown("""
        ```
        ┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐    ┌─────────────┐
        │  PARSE   │ →  │  FEATURES    │ →  │  SCORING   │ →  │  TREE BUILD  │ →  │  VECTORIZE  │
        │ PyMuPDF  │    │ VDS,Td,Lc    │    │ CS vs τ    │    │ C++ N-ary    │    │ HNSW Index  │
        └──────────┘    └──────────────┘    └────────────┘    └──────────────┘    └─────────────┘
        ```
        """)

        st.markdown("---")

        log_text = "\n".join(st.session_state.processing_log)
        st.code(log_text, language="log")

        if st.session_state.ingest_stats:
            st.markdown("### ⏱️ Pipeline Summary")
            st.metric("Total Pipeline Latency", f"{st.session_state.ingest_stats['total_latency_ms']:.1f} ms")
    else:
        st.info("No processing log yet. Upload or load a document to see the pipeline trace.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: Raw Data Inspector
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("📜 Raw Data Inspector")

    choice = st.radio(
        "Select data to inspect:",
        ["Tree JSON", "All Node Payloads", "Page Features", "HNSW Config"],
        horizontal=True
    )

    if choice == "Tree JSON":
        st.json(st.session_state.current_tree.to_dict())

    elif choice == "All Node Payloads":
        st.json(st.session_state.current_tree.get_all_nodes())

    elif choice == "Page Features":
        st.json(st.session_state.page_features)

    elif choice == "HNSW Config":
        st.json({
            "collection_name": "amhie_streamlit",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "vector_dimensions": 384,
            "distance_metric": "Cosine",
            "hnsw_m": 16,
            "hnsw_ef_construct": 100,
            "indexed_points": st.session_state.ingest_stats["indexed_vectors"],
            "tree_nodes_total": st.session_state.ingest_stats["tree_nodes"],
        })
