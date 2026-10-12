"""
Python wrapper for the AMHIE C++17 Structural N-ary Tree Engine.
Supports dynamic loading via ctypes, Pybind11 compiled extensions,
and transparent pure-Python fallback for maximum platform portability.
"""

import os
import sys
import json
import ctypes
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from .logger import safe_print


@dataclass
class TreeNode:
    id: str
    node_type: str  # DOCUMENT, PAGE_COARSE, PAGE_COMPLEX, TEXT, TABLE, IMAGE
    content: str
    page_number: int
    bounding_box: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    metadata: Dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children: List['TreeNode'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "page_number": self.page_number,
            "content": self.content,
            "bounding_box": self.bounding_box,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children]
        }


class _PurePythonTree:
    """Internal fallback tree implementation when C++ dynamic library is not present."""
    def __init__(self, doc_id: str = "doc_root", doc_title: str = "Document"):
        self.root = TreeNode(
            id=doc_id,
            node_type="DOCUMENT",
            content=doc_title,
            page_number=0,
            bounding_box=[0.0, 0.0, 0.0, 0.0],
            metadata={},
            parent_id=None
        )
        self.lookup: Dict[str, TreeNode] = {doc_id: self.root}

    def add_page_node(self, id: str, node_type: str, content: str, page_number: int,
                      bbox: List[float], metadata: Dict[str, str]) -> TreeNode:
        if id in self.lookup:
            return self.lookup[id]
        node = TreeNode(
            id=id,
            node_type=node_type,
            content=content,
            page_number=page_number,
            bounding_box=bbox,
            metadata=metadata,
            parent_id=self.root.id
        )
        self.root.children.append(node)
        self.lookup[id] = node
        return node

    def add_child_element(self, parent_id: str, id: str, node_type: str, content: str,
                          page_number: int, bbox: List[float], metadata: Dict[str, str]) -> Optional[TreeNode]:
        parent = self.lookup.get(parent_id)
        if not parent:
            return None
        if id in self.lookup:
            return self.lookup[id]
        child = TreeNode(
            id=id,
            node_type=node_type,
            content=content,
            page_number=page_number,
            bounding_box=bbox,
            metadata=metadata,
            parent_id=parent_id
        )
        parent.children.append(child)
        self.lookup[id] = child
        return child

    def traverse(self, order: str = "PREORDER") -> List[TreeNode]:
        result = []
        if order == "BFS":
            queue = [self.root]
            while queue:
                curr = queue.pop(0)
                result.append(curr)
                queue.extend(curr.children)
        else:
            def _dfs(node: TreeNode):
                result.append(node)
                for child in node.children:
                    _dfs(child)
            _dfs(self.root)
        return result

    def export_json(self) -> str:
        return json.dumps(self.root.to_dict(), indent=2)

    def size(self) -> int:
        return len(self.lookup)


class DocumentTree:
    """
    AMHIE Document Tree. Interacts with high-performance C++ NArrayTree
    via ctypes, with pure Python fallback.
    """
    def __init__(self, doc_id: str = "doc_root", doc_title: str = "Document"):
        self.doc_id = doc_id
        self.doc_title = doc_title
        self._c_lib = self._load_cpp_library()
        self._tree_handle = None
        self._pure_python = None

        if self._c_lib:
            try:
                self._tree_handle = self._c_lib.amhie_tree_create(
                    doc_id.encode('utf-8'),
                    doc_title.encode('utf-8')
                )
            except Exception as e:
                safe_print(f"[DocumentTree] Warning: Failed to init C++ tree handle: {e}. Falling back to Python.")
                self._pure_python = _PurePythonTree(doc_id, doc_title)
        else:
            self._pure_python = _PurePythonTree(doc_id, doc_title)

    @staticmethod
    def _load_cpp_library():
        """Locates and loads libtree_engine shared library."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csrc_dir = os.path.normpath(os.path.join(current_dir, "..", "csrc"))

        lib_names = ["libtree_engine.dylib", "libtree_engine.so", "tree_engine.dll"]
        for name in lib_names:
            lib_path = os.path.join(csrc_dir, name)
            if os.path.exists(lib_path):
                try:
                    lib = ctypes.CDLL(lib_path)
                    # Define argtypes and restype
                    lib.amhie_tree_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                    lib.amhie_tree_create.restype = ctypes.c_void_p

                    lib.amhie_tree_destroy.argtypes = [ctypes.c_void_p]
                    lib.amhie_tree_destroy.restype = None

                    lib.amhie_tree_add_page_node.argtypes = [
                        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                        ctypes.c_char_p, ctypes.c_int,
                        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                        ctypes.c_char_p
                    ]
                    lib.amhie_tree_add_page_node.restype = ctypes.c_int

                    lib.amhie_tree_add_child_element.argtypes = [
                        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                        ctypes.c_char_p
                    ]
                    lib.amhie_tree_add_child_element.restype = ctypes.c_int

                    lib.amhie_tree_export_json.argtypes = [ctypes.c_void_p]
                    lib.amhie_tree_export_json.restype = ctypes.c_void_p

                    lib.amhie_tree_free_string.argtypes = [ctypes.c_void_p]
                    lib.amhie_tree_free_string.restype = None

                    lib.amhie_tree_get_size.argtypes = [ctypes.c_void_p]
                    lib.amhie_tree_get_size.restype = ctypes.c_int

                    return lib
                except Exception as e:
                    safe_print(f"[DocumentTree] Failed loading shared library at {lib_path}: {e}")
        return None

    def __del__(self):
        if self._c_lib and self._tree_handle:
            try:
                self._c_lib.amhie_tree_destroy(self._tree_handle)
                self._tree_handle = None
            except Exception:
                pass

    def add_page_node(self,
                      id: str,
                      node_type: str,
                      content: str,
                      page_number: int,
                      bbox: Optional[List[float]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Adds a top-level page node to the document tree."""
        bbox = bbox or [0.0, 0.0, 0.0, 0.0]
        metadata = {str(k): str(v) for k, v in (metadata or {}).items()}

        if self._pure_python is not None:
            self._pure_python.add_page_node(id, node_type, content, page_number, bbox, metadata)
            return True

        meta_json = json.dumps(metadata)
        res = self._c_lib.amhie_tree_add_page_node(
            self._tree_handle,
            id.encode('utf-8'),
            node_type.encode('utf-8'),
            content.encode('utf-8'),
            int(page_number),
            float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
            meta_json.encode('utf-8')
        )
        return bool(res)

    def add_child_element(self,
                          parent_id: str,
                          id: str,
                          node_type: str,
                          content: str,
                          page_number: int,
                          bbox: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Adds a fine-grained element (table, image, text block) to a parent page node."""
        bbox = bbox or [0.0, 0.0, 0.0, 0.0]
        metadata = {str(k): str(v) for k, v in (metadata or {}).items()}

        if self._pure_python is not None:
            res = self._pure_python.add_child_element(parent_id, id, node_type, content, page_number, bbox, metadata)
            return res is not None

        meta_json = json.dumps(metadata)
        res = self._c_lib.amhie_tree_add_child_element(
            self._tree_handle,
            parent_id.encode('utf-8'),
            id.encode('utf-8'),
            node_type.encode('utf-8'),
            content.encode('utf-8'),
            int(page_number),
            float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
            meta_json.encode('utf-8')
        )
        return bool(res)

    def export_json(self) -> str:
        """Exports the entire hierarchical tree to a JSON string."""
        if self._pure_python is not None:
            return self._pure_python.export_json()

        ptr = self._c_lib.amhie_tree_export_json(self._tree_handle)
        if not ptr:
            return "{}"
        try:
            json_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode('utf-8')
            return json_str
        finally:
            self._c_lib.amhie_tree_free_string(ptr)

    def to_dict(self) -> Dict[str, Any]:
        """Returns tree representation as a nested Python dictionary."""
        json_str = self.export_json()
        try:
            return json.loads(json_str)
        except Exception:
            return {}

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Flattens the tree into a list of node dictionaries with parent pointers."""
        root_dict = self.to_dict()
        nodes = []

        def _collect(curr: Dict[str, Any]):
            children = curr.get("children", [])
            node_copy = {k: v for k, v in curr.items() if k != "children"}
            nodes.append(node_copy)
            for child in children:
                _collect(child)

        if root_dict:
            _collect(root_dict)
        return nodes

    def size(self) -> int:
        """Returns total number of nodes in the tree."""
        if self._pure_python is not None:
            return self._pure_python.size()
        return int(self._c_lib.amhie_tree_get_size(self._tree_handle))

    def __len__(self) -> int:
        return self.size()
