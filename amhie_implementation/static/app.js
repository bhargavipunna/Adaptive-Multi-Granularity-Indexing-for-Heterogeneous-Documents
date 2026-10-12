/**
 * AMHIE: Adaptive Multi-Granularity Hierarchical Indexing Engine
 * Frontend Client Application
 * Handles interactive tabs, C++ tree rendering, storage inspection,
 * live vector point exploration, and adaptive RAG queries.
 */

// Application State
const state = {
  activeTab: "overview",
  activeDocId: null,
  activeDocMeta: null,
  treeData: null,
  selectedNode: null,
  vectorPoints: [],
  filteredVectors: [],
  threshold: 0.45,
  isLoading: false,
};

// ==============================================================================
// Initialization
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
  initLucide();
  setupTabNavigation();
  setupEventListeners();
  loadInitialData();
});

function initLucide() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

// ==============================================================================
// Tab Navigation
// ==============================================================================
function setupTabNavigation() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  state.activeTab = tabId;

  // Update button active classes
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    if (btn.getAttribute("data-tab") === tabId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update tab views
  document.querySelectorAll(".tab-view").forEach((view) => {
    if (view.id === `view-${tabId}`) {
      view.classList.remove("hidden");
      view.classList.add("active");
    } else {
      view.classList.add("hidden");
      view.classList.remove("active");
    }
  });

  // Lazy load or refresh specific tab content
  if (tabId === "tree" && state.activeDocId && !state.treeData) {
    fetchTreeGraph(state.activeDocId);
  } else if (tabId === "storage") {
    fetchVectorPoints();
  }

  initLucide();
}

// ==============================================================================
// Event Listeners & UI Controls
// ==============================================================================
function setupEventListeners() {
  // Sample document load button
  const loadSampleBtn = document.getElementById("btn-load-sample");
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener("click", () => loadSampleBenchmark());
  }

  // Refresh overview
  const refreshOverviewBtn = document.getElementById("btn-refresh-overview");
  if (refreshOverviewBtn) {
    refreshOverviewBtn.addEventListener("click", () => loadInitialData());
  }

  // Threshold slider
  const slider = document.getElementById("slider-tau");
  const sliderVal = document.getElementById("slider-tau-val");
  if (slider && sliderVal) {
    slider.addEventListener("input", (e) => {
      state.threshold = parseFloat(e.target.value);
      sliderVal.textContent = state.threshold.toFixed(2);
    });
  }

  // Re-index with current threshold
  const reindexBtn = document.getElementById("btn-reindex-sample");
  if (reindexBtn) {
    reindexBtn.addEventListener("click", () => {
      loadSampleBenchmark(state.threshold);
    });
  }

  // File Upload Dropzone
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadDocument(e.target.files[0]);
      }
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("border-cyan-500");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("border-cyan-500");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("border-cyan-500");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadDocument(e.dataTransfer.files[0]);
      }
    });
  }

  // RAG Query Execution
  const runRagBtn = document.getElementById("btn-run-rag");
  const ragInput = document.getElementById("rag-query-input");
  if (runRagBtn && ragInput) {
    runRagBtn.addEventListener("click", () => {
      const q = ragInput.value.trim();
      if (q) executeRagQuery(q);
    });

    ragInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const q = ragInput.value.trim();
        if (q) executeRagQuery(q);
      }
    });
  }

  // Preset query chips
  document.querySelectorAll(".query-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const queryText = chip.textContent.replace(/^[^\w]+/, "").trim();
      if (ragInput) ragInput.value = queryText;
      executeRagQuery(queryText);
    });
  });

  // Vector Search Filter
  const vectorSearchInput = document.getElementById("search-vectors-input");
  if (vectorSearchInput) {
    vectorSearchInput.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase();
      state.filteredVectors = state.vectorPoints.filter((p) => {
        return (
          (p.node_id && p.node_id.toLowerCase().includes(q)) ||
          (p.node_type && p.node_type.toLowerCase().includes(q)) ||
          (p.content_preview && p.content_preview.toLowerCase().includes(q)) ||
          (p.parent_id && p.parent_id.toLowerCase().includes(q))
        );
      });
      renderVectorTable(state.filteredVectors);
    });
  }

  // Modal Close
  const closeModalBtn = document.getElementById("btn-close-modal");
  const modal = document.getElementById("vector-detail-modal");
  if (closeModalBtn && modal) {
    closeModalBtn.addEventListener("click", () => modal.classList.add("hidden"));
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.add("hidden");
    });
  }

  // Tree filter buttons
  document.getElementById("btn-filter-all")?.addEventListener("click", () => filterTreeNodes("ALL"));
  document.getElementById("btn-filter-tables")?.addEventListener("click", () => filterTreeNodes("TABLE"));
  document.getElementById("btn-filter-complex")?.addEventListener("click", () => filterTreeNodes("PAGE_COMPLEX"));
}

// ==============================================================================
// Data Fetching & API Communication
// ==============================================================================
async function loadInitialData() {
  try {
    const res = await fetch("/api/storage/overview");
    if (!res.ok) throw new Error("Failed to connect to AMHIE storage service.");
    const data = await res.json();

    state.activeDocId = data.active_document_id;
    state.activeDocMeta = data.active_document;

    updateOverviewUI(data);

    if (state.activeDocId) {
      fetchTreeGraph(state.activeDocId);
    }
    fetchVectorPoints();
  } catch (err) {
    console.error("[AMHIE App] Error loading initial data:", err);
    showToast("Connecting to backend engine...", "info");
  }
}

async function loadSampleBenchmark(threshold = 0.45) {
  showToast("Indexing heterogeneous benchmark document...", "info");
  try {
    const res = await fetch(`/api/documents/sample?threshold=${threshold}`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to load sample document.");
    const data = await res.json();

    state.activeDocId = data.doc_id;
    state.activeDocMeta = data;
    showToast(`Successfully indexed '${data.filename}' with ${data.total_nodes_in_tree} logical nodes!`, "success");

    loadInitialData();
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
  }
}

async function uploadDocument(file) {
  showToast(`Uploading and parsing '${file.name}'...`, "info");
  const formData = new FormData();
  formData.append("file", file);
  formData.append("threshold", state.threshold.toString());

  try {
    const res = await fetch("/api/documents/upload", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Upload failed");
    }
    const data = await res.json();
    state.activeDocId = data.doc_id;
    state.activeDocMeta = data;
    showToast(`Successfully parsed and indexed '${data.filename}'!`, "success");
    loadInitialData();
  } catch (err) {
    showToast(`Upload Error: ${err.message}`, "error");
  }
}

// ==============================================================================
// UI Updates: Overview & KPI Telemetry
// ==============================================================================
function updateOverviewUI(overview) {
  const doc = overview.active_document;
  if (!doc) return;

  // Update Hero Banner
  document.getElementById("hero-doc-title").textContent = doc.filename || "Uploaded Document";
  document.getElementById("hero-doc-subtitle").textContent = `ID: ${doc.doc_id} | ${doc.total_pages} Pages | Threshold: Tau=${doc.threshold_used || 0.45}`;
  document.getElementById("hero-storage-savings").textContent = `${doc.storage_reduction_pct || 66.7}% vector reduction`;

  // Update KPI Telemetry Cards
  animateCounter("kpi-total-nodes", doc.total_nodes_in_tree || 0);
  animateCounter("kpi-total-vectors", doc.total_vectors_indexed || 0);
  animateCounter("kpi-coarse-pages", doc.coarse_pages || 0);
  animateCounter("kpi-complex-pages", doc.complex_pages || 0);
  document.getElementById("kpi-savings-pct").textContent = `${doc.storage_reduction_pct || 66.7}%`;
  document.getElementById("storage-vector-badge").textContent = `${doc.total_vectors_indexed || 0} Vectors`;

  // Populate Per-Page Complexity Telemetry Table
  const tableBody = document.getElementById("table-page-breakdown");
  if (tableBody && doc.pages) {
    tableBody.innerHTML = doc.pages
      .map((p) => {
        const isComplex = p.decision === "PAGE_COMPLEX";
        const badgeClass = isComplex
          ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
          : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
        const decisionText = isComplex ? "COMPLEX (N-ary Subtree)" : "COARSE (Parent Node)";

        return `
        <tr class="hover:bg-dark-hover transition-colors">
          <td class="p-3 text-white font-bold">Page ${p.page_number}</td>
          <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] font-semibold border ${badgeClass}">${decisionText}</span></td>
          <td class="p-3 text-cyan-400 font-bold">${p.complexity_score.toFixed(4)}</td>
          <td class="p-3 text-slate-300">${p.vds.toFixed(4)}</td>
          <td class="p-3 text-slate-300">${p.token_density.toFixed(1)}</td>
          <td class="p-3 text-cyan-300">${p.tables_count || 0}</td>
          <td class="p-3 text-violet-300">${p.images_count || 0}</td>
          <td class="p-3 text-slate-400">${p.words_count || 0}</td>
        </tr>
      `;
      })
      .join("");
  }
}

function animateCounter(elemId, targetVal) {
  const elem = document.getElementById(elemId);
  if (!elem) return;
  let start = 0;
  const duration = 800;
  const stepTime = 20;
  const steps = duration / stepTime;
  const increment = targetVal / steps;

  const timer = setInterval(() => {
    start += increment;
    if (start >= targetVal) {
      elem.textContent = targetVal;
      clearInterval(timer);
    } else {
      elem.textContent = Math.floor(start);
    }
  }, stepTime);
}

// ==============================================================================
// C++ N-ary Tree Graph Visualization
// ==============================================================================
async function fetchTreeGraph(docId) {
  try {
    const res = await fetch(`/api/tree/${docId}`);
    if (!res.ok) throw new Error("Could not fetch document tree.");
    const tree = await res.json();
    state.treeData = tree;
    renderTreeGraph(tree);
  } catch (err) {
    console.error("[TreeGraph] Error fetching tree:", err);
  }
}

function renderTreeGraph(tree) {
  const container = document.getElementById("tree-graph-canvas");
  if (!container) return;

  container.innerHTML = "";
  const rootNode = tree;

  // Recursive Tree Node Builder
  function buildNodeHTML(node, depth = 0) {
    const isRoot = node.node_type === "DOCUMENT";
    const isCoarse = node.node_type === "PAGE_COARSE";
    const isComplex = node.node_type === "PAGE_COMPLEX";
    const isTable = node.node_type === "TABLE";
    const isImage = node.node_type === "IMAGE";
    const isText = node.node_type === "TEXT";

    let borderCol = "border-dark-border";
    let badgeCol = "bg-slate-800 text-slate-300 border-slate-700";
    let icon = "file";

    if (isRoot) {
      borderCol = "border-indigo-500/40";
      badgeCol = "bg-indigo-500/20 text-indigo-300 border-indigo-500/30";
      icon = "book-open";
    } else if (isCoarse) {
      borderCol = "border-emerald-500/40";
      badgeCol = "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
      icon = "file-text";
    } else if (isComplex) {
      borderCol = "border-amber-500/40";
      badgeCol = "bg-amber-500/20 text-amber-300 border-amber-500/30";
      icon = "folder-tree";
    } else if (isTable) {
      borderCol = "border-cyan-500/40";
      badgeCol = "bg-cyan-500/20 text-cyan-300 border-cyan-500/30";
      icon = "table";
    } else if (isImage) {
      borderCol = "border-violet-500/40";
      badgeCol = "bg-violet-500/20 text-violet-300 border-violet-500/30";
      icon = "image";
    }

    const paddingLeft = depth * 24;
    const hasChildren = node.children && node.children.length > 0;
    const cs = node.metadata?.complexity_score ? ` (CS: ${node.metadata.complexity_score})` : "";

    const card = document.createElement("div");
    card.className = `tree-node-item my-1.5`;
    card.style.paddingLeft = `${paddingLeft}px`;

    card.innerHTML = `
      <div data-node-id="${node.id}" class="tree-node-card cursor-pointer p-3 rounded-xl bg-dark-surface/90 border ${borderCol} hover:bg-dark-hover flex items-center justify-between transition-all">
        <div class="flex items-center space-x-3 overflow-hidden">
          <div class="p-1.5 rounded-lg ${badgeCol}">
            <i data-lucide="${icon}" class="w-4 h-4"></i>
          </div>
          <div class="truncate">
            <div class="flex items-center space-x-2">
              <span class="font-mono text-xs font-bold text-white">${node.id}</span>
              <span class="px-1.5 py-0.2 rounded text-[10px] font-mono border ${badgeCol}">${node.node_type}${cs}</span>
            </div>
            <p class="text-[11px] text-slate-400 truncate max-w-sm mt-0.5">${node.content ? node.content.slice(0, 80) : ""}</p>
          </div>
        </div>
        <div class="flex items-center space-x-2 text-[11px] font-mono text-slate-400">
          <span>P.${node.page_number}</span>
          ${hasChildren ? `<span class="px-1.5 py-0.5 rounded bg-dark-card border border-dark-border text-[10px]">${node.children.length} sub</span>` : ""}
        </div>
      </div>
    `;

    card.querySelector(".tree-node-card").addEventListener("click", () => {
      document.querySelectorAll(".tree-node-card").forEach((c) => c.classList.remove("selected"));
      card.querySelector(".tree-node-card").classList.add("selected");
      inspectNode(node);
    });

    container.appendChild(card);

    if (hasChildren) {
      node.children.forEach((child) => buildNodeHTML(child, depth + 1));
    }
  }

  buildNodeHTML(rootNode, 0);
  initLucide();

  // Select root by default
  inspectNode(rootNode);
}

function filterTreeNodes(type) {
  document.querySelectorAll(".filter-node-btn").forEach((btn) => btn.classList.remove("active", "bg-cyan-500", "text-white"));
  document.querySelectorAll(".filter-node-btn").forEach((btn) => btn.classList.add("bg-dark-card", "text-slate-300"));

  if (type === "ALL") {
    document.getElementById("btn-filter-all")?.classList.add("active", "bg-cyan-500", "text-white");
    document.querySelectorAll(".tree-node-item").forEach((el) => el.classList.remove("hidden"));
  } else {
    const btn = type === "TABLE" ? document.getElementById("btn-filter-tables") : document.getElementById("btn-filter-complex");
    btn?.classList.add("active", "bg-cyan-500", "text-white");

    document.querySelectorAll(".tree-node-item").forEach((el) => {
      const isMatch = el.innerHTML.includes(type);
      if (isMatch) el.classList.remove("hidden");
      else el.classList.add("hidden");
    });
  }
}

function inspectNode(node) {
  state.selectedNode = node;
  document.getElementById("insp-node-id").textContent = node.id || "N/A";
  document.getElementById("insp-node-page").textContent = node.page_number ?? 0;
  document.getElementById("insp-node-parent").textContent = node.parent_id || "None (Root)";
  document.getElementById("insp-node-bbox").textContent = JSON.stringify(node.bounding_box || [0, 0, 0, 0]);
  document.getElementById("inspector-node-type-badge").textContent = node.node_type || "UNKNOWN";

  const contentElem = document.getElementById("insp-node-text");
  contentElem.textContent = node.content || "No textual content available.";

  const metaElem = document.getElementById("insp-node-meta");
  metaElem.textContent = JSON.stringify(node.metadata || {}, null, 2);
}

// ==============================================================================
// Vector DB Point Inspector Table
// ==============================================================================
async function fetchVectorPoints() {
  try {
    const res = await fetch("/api/storage/vectors?limit=100");
    if (!res.ok) throw new Error("Could not retrieve vector store points.");
    const points = await res.json();
    state.vectorPoints = points;
    state.filteredVectors = points;
    renderVectorTable(points);
  } catch (err) {
    console.error("[Vectors] Error loading vectors:", err);
  }
}

function renderVectorTable(points) {
  const tbody = document.getElementById("table-vectors-body");
  if (!tbody) return;

  if (points.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="p-6 text-center text-slate-500 font-sans">No indexed vector points found in storage.</td></tr>`;
    return;
  }

  tbody.innerHTML = points
    .map((p, idx) => {
      const typeBadgeCol =
        p.node_type === "TABLE"
          ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/30"
          : p.node_type === "IMAGE"
          ? "bg-violet-500/20 text-violet-300 border-violet-500/30"
          : p.node_type === "PAGE_COMPLEX"
          ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
          : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";

      const vecSampleStr = p.vector_sample ? `[${p.vector_sample.slice(0, 4).join(", ")}...]` : "[384 floats]";

      return `
      <tr class="hover:bg-dark-hover transition-colors">
        <td class="p-3 text-cyan-400 font-bold">${p.point_id || p.node_id}</td>
        <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] font-semibold border ${typeBadgeCol}">${p.node_type}</span></td>
        <td class="p-3 text-slate-300">P.${p.page_number}</td>
        <td class="p-3 text-slate-400">${p.parent_id || "-"}</td>
        <td class="p-3 text-violet-400 font-mono text-[11px]">${vecSampleStr}</td>
        <td class="p-3 text-slate-300 font-sans text-xs truncate max-w-xs">${p.content_preview}</td>
        <td class="p-3 text-right">
          <button onclick="openVectorModal(${idx})" class="px-2.5 py-1 rounded bg-dark-card hover:bg-dark-surface border border-dark-border text-cyan-300 text-[11px] font-semibold transition-all">
            Inspect Payload
          </button>
        </td>
      </tr>
    `;
    })
    .join("");
}

window.openVectorModal = function (idx) {
  const point = state.filteredVectors[idx];
  if (!point) return;

  document.getElementById("modal-point-title").textContent = `Vector Point: ${point.node_id} (${point.node_type})`;
  document.getElementById("modal-payload-json").textContent = JSON.stringify(
    {
      node_id: point.node_id,
      node_type: point.node_type,
      page_number: point.page_number,
      parent_id: point.parent_id,
      is_coarse: point.is_coarse,
      complexity_score: point.complexity_score,
      bounding_box: point.bounding_box,
      metadata: point.metadata,
    },
    null,
    2
  );

  document.getElementById("modal-vector-sample").textContent = JSON.stringify(
    {
      dimensions: point.vector_dimensions || 384,
      embedding_model: "BAAI/bge-small-en-v1.5",
      vector_head_8: point.vector_sample,
    },
    null,
    2
  );

  document.getElementById("vector-detail-modal").classList.remove("hidden");
  initLucide();
};

// ==============================================================================
// Adaptive RAG Execution & Traversal Trace
// ==============================================================================
async function executeRagQuery(query) {
  const answerElem = document.getElementById("rag-answer-content");
  const latencyBadge = document.getElementById("rag-latency-badge");
  const generatorBadge = document.getElementById("rag-generator-badge");
  const pathsContainer = document.getElementById("rag-traversal-paths");
  const contextsGrid = document.getElementById("rag-contexts-grid");
  const countBadge = document.getElementById("rag-contexts-count");

  answerElem.innerHTML = `<div class="flex items-center space-x-2 text-cyan-400"><i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Executing adaptive two-stage HNSW retrieval & synthesizing response...</span></div>`;
  initLucide();

  try {
    const res = await fetch("/api/rag/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 3 }),
    });

    if (!res.ok) throw new Error("Failed to execute RAG query.");
    const data = await res.json();

    // Render Answer with Markdown
    if (window.marked) {
      answerElem.innerHTML = window.marked.parse(data.answer);
    } else {
      answerElem.textContent = data.answer;
    }

    // Latency & Model
    latencyBadge.textContent = `Latency: ${data.latency_ms.toFixed(1)} ms`;
    generatorBadge.textContent = data.model_used;

    // Render Traversal Paths
    if (data.retrieval_paths && data.retrieval_paths.length > 0) {
      pathsContainer.innerHTML = data.retrieval_paths
        .map(
          (path, i) => `
        <div class="p-2.5 rounded-xl bg-dark-surface border border-dark-border flex items-center space-x-2">
          <span class="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-bold text-[10px]">PATH ${i + 1}</span>
          <span class="text-slate-300 font-mono text-xs">${path}</span>
        </div>
      `
        )
        .join("");
    } else {
      pathsContainer.innerHTML = `<div class="p-3 text-slate-500">Direct coarse match (single-stage retrieval).</div>`;
    }

    // Render Contexts
    countBadge.textContent = `${data.contexts.length} contexts retrieved`;
    contextsGrid.innerHTML = data.contexts
      .map((c) => {
        const typeBadgeCol =
          c.node_type === "TABLE"
            ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/30"
            : c.node_type === "PAGE_COMPLEX"
            ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
            : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";

        const contentHtml = window.marked ? window.marked.parse(c.content) : c.content;

        return `
        <div class="p-4 rounded-xl bg-dark-surface border border-dark-border space-y-2">
          <div class="flex items-center justify-between text-xs">
            <span class="px-2 py-0.5 rounded font-mono border ${typeBadgeCol}">${c.node_type}</span>
            <span class="font-mono text-cyan-400">Score: ${c.similarity_score.toFixed(4)}</span>
          </div>
          <div class="flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>Node: ${c.node_id}</span>
            <span>Page ${c.page_number}</span>
          </div>
          <div class="pt-2 border-t border-dark-border text-xs text-slate-300 max-h-48 overflow-y-auto">
            ${contentHtml}
          </div>
        </div>
      `;
      })
      .join("");

    initLucide();
  } catch (err) {
    answerElem.innerHTML = `<div class="text-rose-400">Error executing RAG: ${err.message}</div>`;
  }
}

// ==============================================================================
// Toast Notifications
// ==============================================================================
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  let bgCol = "bg-dark-card border-dark-border text-slate-200";
  if (type === "success") bgCol = "bg-emerald-950/90 border-emerald-500/50 text-emerald-200";
  if (type === "error") bgCol = "bg-rose-950/90 border-rose-500/50 text-rose-200";

  toast.className = `toast-msg p-3.5 rounded-xl border shadow-xl flex items-center space-x-2 text-xs font-medium ${bgCol}`;
  toast.innerHTML = `
    <span class="w-2 h-2 rounded-full ${type === "success" ? "bg-emerald-400" : type === "error" ? "bg-rose-400" : "bg-cyan-400"}"></span>
    <span>${message}</span>
  `;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
