import React, { useState } from 'react';
import { Play, Sparkles, Compass, Layers } from 'lucide-react';

export interface RetrievedContextItem {
  node_id: string;
  node_type: string;
  page_number: number;
  content: string;
  similarity_score: number;
  bounding_box: number[];
  traversal_path: string;
}

export interface RagResponseData {
  query: string;
  answer: string;
  contexts: RetrievedContextItem[];
  retrieval_paths: string[];
  latency_ms: number;
  model_used: string;
}

export const RagConsole: React.FC<{
  onExecute: (query: string) => Promise<RagResponseData>;
}> = ({ onExecute }) => {
  const [query, setQuery] = useState('What were the Q4 2024 operating margins and revenue?');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RagResponseData | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await onExecute(query);
      setResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
        <label className="text-xs font-bold text-slate-300">Document Question</label>
        <div className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 px-4 py-3 rounded-xl bg-slate-950 border border-slate-800 text-sm text-white focus:outline-none focus:border-cyan-500"
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-emerald-600 hover:from-cyan-500 hover:to-emerald-500 text-white font-semibold text-sm flex items-center space-x-2"
          >
            <Play className="w-4 h-4" />
            <span>{loading ? 'Searching...' : 'Search'}</span>
          </button>
        </div>
      </div>

      {result && (
        <div className="space-y-6">
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-cyan-500/30 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-cyan-400 flex items-center space-x-1.5">
                <Sparkles className="w-4 h-4" />
                <span>SYNTHESIZED ANSWER</span>
              </span>
              <span className="text-xs font-mono text-emerald-400">
                {result.latency_ms.toFixed(1)} ms
              </span>
            </div>
            <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
              {result.answer}
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
            <h4 className="text-xs font-bold text-slate-400 flex items-center space-x-1.5">
              <Compass className="w-4 h-4 text-amber-400" />
              <span>TRAVERSAL TRACE</span>
            </h4>
            <div className="space-y-1.5 font-mono text-xs text-slate-300">
              {result.retrieval_paths.map((p, idx) => (
                <div key={idx} className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  {p}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
