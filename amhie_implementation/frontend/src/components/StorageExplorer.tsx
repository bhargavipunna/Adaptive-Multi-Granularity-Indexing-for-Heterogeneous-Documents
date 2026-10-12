import React, { useState } from 'react';
import { Database, Cpu, Layers, HardDrive, Search, ExternalLink } from 'lucide-react';

export interface VectorPoint {
  point_id: string;
  node_id: string;
  node_type: string;
  page_number: number;
  parent_id?: string;
  is_coarse: boolean;
  complexity_score: number;
  content_preview: string;
  bounding_box: number[];
  vector_dimensions: number;
  vector_sample: number[];
  metadata: Record<string, any>;
}

export interface StorageOverviewData {
  status: string;
  active_document_id: string;
  storage_tiers: {
    tier_1_multimodal: any;
    tier_2_cpp_tree: any;
    tier_3_qdrant_hnsw: any;
    tier_4_complexity_store: any;
  };
  storage_economy: {
    estimated_flat_chunks: number;
    actual_amhie_vectors: number;
    reduction_percentage: number;
    table_structural_integrity: string;
  };
}

export const StorageExplorer: React.FC<{
  overview: StorageOverviewData | null;
  vectors: VectorPoint[];
}> = ({ overview, vectors }) => {
  const [search, setSearch] = useState('');
  const [selectedPoint, setSelectedPoint] = useState<VectorPoint | null>(null);

  const filtered = vectors.filter(
    (v) =>
      v.node_id?.toLowerCase().includes(search.toLowerCase()) ||
      v.node_type?.toLowerCase().includes(search.toLowerCase()) ||
      v.content_preview?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* 4 Storage Tiers Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Tier 1 */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-cyan-500/40 transition-all">
          <div className="flex items-center justify-between mb-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300">
              TIER 1
            </span>
            <HardDrive className="w-4 h-4 text-cyan-400" />
          </div>
          <h4 className="text-sm font-bold text-white">Multimodal Parse Store</h4>
          <p className="text-xs text-slate-400 mt-1">PyMuPDF / MinerU extraction of tables and bounding boxes.</p>
          <div className="mt-3 pt-2 border-t border-slate-800 text-[11px] font-mono text-slate-300 space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Format:</span>
              <span>Markdown Tables</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Integrity:</span>
              <span className="text-emerald-400">100% Preserved</span>
            </div>
          </div>
        </div>

        {/* Tier 2 */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-emerald-500/40 transition-all">
          <div className="flex items-center justify-between mb-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300">
              TIER 2
            </span>
            <Cpu className="w-4 h-4 text-emerald-400" />
          </div>
          <h4 className="text-sm font-bold text-white">C++17 Tree Memory</h4>
          <p className="text-xs text-slate-400 mt-1">Compiled N-ary pointer hierarchy with sub-ms tree traversal.</p>
          <div className="mt-3 pt-2 border-t border-slate-800 text-[11px] font-mono text-slate-300 space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Library:</span>
              <span>libtree_engine.dylib</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Structure:</span>
              <span>std::shared_ptr</span>
            </div>
          </div>
        </div>

        {/* Tier 3 */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-violet-500/40 transition-all">
          <div className="flex items-center justify-between mb-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-violet-500/20 text-violet-300">
              TIER 3
            </span>
            <Database className="w-4 h-4 text-violet-400" />
          </div>
          <h4 className="text-sm font-bold text-white">Qdrant Vector DB</h4>
          <p className="text-xs text-slate-400 mt-1">384-dimensional dense vectors with HNSW graph indexing.</p>
          <div className="mt-3 pt-2 border-t border-slate-800 text-[11px] font-mono text-slate-300 space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Index:</span>
              <span className="text-violet-400">HNSW (M=16, ef=100)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Metric:</span>
              <span>Cosine</span>
            </div>
          </div>
        </div>

        {/* Tier 4 */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-amber-500/40 transition-all">
          <div className="flex items-center justify-between mb-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300">
              TIER 4
            </span>
            <Layers className="w-4 h-4 text-amber-400" />
          </div>
          <h4 className="text-sm font-bold text-white">Complexity Feature Store</h4>
          <p className="text-xs text-slate-400 mt-1">Continuous scoring (VDS, Td, Lc) classifying page complexity.</p>
          <div className="mt-3 pt-2 border-t border-slate-800 text-[11px] font-mono text-slate-300 space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Boundary:</span>
              <span className="text-amber-400">Tau = 0.45</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">VDS Weight:</span>
              <span>0.35</span>
            </div>
          </div>
        </div>
      </div>

      {/* Vector Store Explorer */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 backdrop-blur-md space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <Database className="w-4 h-4 text-cyan-400" />
              <span>Live Qdrant Vector Store Point Inspector</span>
            </h3>
            <p className="text-xs text-slate-400">Inspect actual points, payloads, and embeddings stored in vector memory.</p>
          </div>
          <div className="relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search points..."
              className="pl-9 pr-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-white focus:outline-none focus:border-cyan-500"
            />
          </div>
        </div>

        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-left text-xs text-slate-300 font-mono">
            <thead className="bg-slate-950 sticky top-0 text-slate-400 uppercase text-[10px]">
              <tr>
                <th className="p-3">Node ID</th>
                <th className="p-3">Type</th>
                <th className="p-3">Page</th>
                <th className="p-3">Parent</th>
                <th className="p-3">Vector Preview</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {filtered.map((pt) => (
                <tr key={pt.node_id} className="hover:bg-slate-800/50">
                  <td className="p-3 text-cyan-400 font-bold">{pt.node_id}</td>
                  <td className="p-3">
                    <span className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300">
                      {pt.node_type}
                    </span>
                  </td>
                  <td className="p-3">P.{pt.page_number}</td>
                  <td className="p-3 text-slate-400">{pt.parent_id || '-'}</td>
                  <td className="p-3 text-violet-400 text-[11px]">
                    [{pt.vector_sample?.slice(0, 3).join(', ')}...]
                  </td>
                  <td className="p-3 text-right">
                    <button
                      onClick={() => setSelectedPoint(pt)}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 text-xs"
                    >
                      View Payload
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
