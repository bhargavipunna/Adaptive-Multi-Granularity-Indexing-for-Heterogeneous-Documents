import React, { useState } from 'react';
import { BookOpen, FileText, FolderTree, Table, Image, Cpu } from 'lucide-react';

export interface TreeNodeData {
  id: string;
  node_type: string;
  page_number: number;
  content: string;
  bounding_box: number[];
  metadata: Record<string, any>;
  parent_id?: string;
  children?: TreeNodeData[];
}

export const TreeVisualizer: React.FC<{
  tree: TreeNodeData | null;
  onSelectNode: (node: TreeNodeData) => void;
}> = ({ tree, onSelectNode }) => {
  const [selectedId, setSelectedId] = useState<string | null>(tree?.id || null);

  if (!tree) {
    return <div className="p-8 text-center text-slate-500">No active document tree loaded.</div>;
  }

  const renderNode = (node: TreeNodeData, depth: number = 0) => {
    const isRoot = node.node_type === 'DOCUMENT';
    const isCoarse = node.node_type === 'PAGE_COARSE';
    const isComplex = node.node_type === 'PAGE_COMPLEX';
    const isTable = node.node_type === 'TABLE';
    const isImage = node.node_type === 'IMAGE';

    let icon = <FileText className="w-4 h-4 text-slate-400" />;
    let badgeClass = 'bg-slate-800 text-slate-300 border-slate-700';

    if (isRoot) {
      icon = <BookOpen className="w-4 h-4 text-indigo-400" />;
      badgeClass = 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30';
    } else if (isCoarse) {
      icon = <FileText className="w-4 h-4 text-emerald-400" />;
      badgeClass = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
    } else if (isComplex) {
      icon = <FolderTree className="w-4 h-4 text-amber-400" />;
      badgeClass = 'bg-amber-500/20 text-amber-300 border-amber-500/30';
    } else if (isTable) {
      icon = <Table className="w-4 h-4 text-cyan-400" />;
      badgeClass = 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30';
    } else if (isImage) {
      icon = <Image className="w-4 h-4 text-violet-400" />;
      badgeClass = 'bg-violet-500/20 text-violet-300 border-violet-500/30';
    }

    const isSelected = selectedId === node.id;

    return (
      <div key={node.id} style={{ paddingLeft: `${depth * 20}px` }} className="my-1.5">
        <div
          onClick={() => {
            setSelectedId(node.id);
            onSelectNode(node);
          }}
          className={`cursor-pointer p-3 rounded-xl border transition-all flex items-center justify-between ${
            isSelected
              ? 'bg-cyan-950/30 border-cyan-500 shadow-md shadow-cyan-500/10'
              : 'bg-slate-900/80 border-slate-800 hover:bg-slate-800/60'
          }`}
        >
          <div className="flex items-center space-x-3 truncate">
            <div className="p-1.5 rounded-lg bg-slate-950">{icon}</div>
            <div className="truncate">
              <div className="flex items-center space-x-2">
                <span className="font-mono text-xs font-bold text-white">{node.id}</span>
                <span className={`px-1.5 py-0.2 rounded text-[10px] font-mono border ${badgeClass}`}>
                  {node.node_type}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 truncate max-w-sm mt-0.5">
                {node.content?.slice(0, 70)}
              </p>
            </div>
          </div>

          <div className="text-[11px] font-mono text-slate-400">
            <span>P.{node.page_number}</span>
          </div>
        </div>

        {node.children && node.children.map((child) => renderNode(child, depth + 1))}
      </div>
    );
  };

  return <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">{renderNode(tree, 0)}</div>;
};
