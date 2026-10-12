"""
Hierarchical Tree Router for AMHIE.
Exposes serialized C++17 N-ary document trees, node inspector, and graph hierarchy.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional

from ..services.service_container import get_service_container

router = APIRouter(prefix="/api/tree", tags=["C++ Structural N-ary Tree"])


@router.get("/{doc_id}")
def get_tree_hierarchy(doc_id: str) -> Dict[str, Any]:
    """Returns the full hierarchical N-ary tree for frontend rendering."""
    container = get_service_container()
    tree = container.active_trees.get(doc_id)
    if not tree:
        raise HTTPException(status_code=404, detail=f"Tree for document '{doc_id}' not found.")
    return tree.to_dict()


@router.get("/{doc_id}/flat")
def get_flat_nodes(doc_id: str) -> List[Dict[str, Any]]:
    """Returns all nodes as a flat list with parent/child links for table and search views."""
    container = get_service_container()
    tree = container.active_trees.get(doc_id)
    if not tree:
        raise HTTPException(status_code=404, detail=f"Tree for document '{doc_id}' not found.")
    return tree.get_all_nodes()


@router.get("/{doc_id}/nodes/{node_id}")
def get_node_details(doc_id: str, node_id: str) -> Dict[str, Any]:
    """Inspects a specific node's memory attributes, bounding box, and content."""
    container = get_service_container()
    tree = container.active_trees.get(doc_id)
    if not tree:
        raise HTTPException(status_code=404, detail=f"Tree for document '{doc_id}' not found.")

    nodes = tree.get_all_nodes()
    match = next((n for n in nodes if n.get("id") == node_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in tree '{doc_id}'.")

    # Find children
    children = [n for n in nodes if n.get("parent_id") == node_id]

    return {
        "doc_id": doc_id,
        "node": match,
        "children_count": len(children),
        "children": children
    }
