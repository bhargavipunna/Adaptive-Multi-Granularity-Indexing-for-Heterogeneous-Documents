#include "tree_engine.h"
#include <queue>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

namespace amhie {

static std::string escape_json_string(const std::string& input) {
    std::ostringstream ss;
    for (char c : input) {
        switch (c) {
            case '"': ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b"; break;
            case '\f': ss << "\\f"; break;
            case '\n': ss << "\\n"; break;
            case '\r': ss << "\\r"; break;
            case '\t': ss << "\\t"; break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    ss << c;
                }
        }
    }
    return ss.str();
}

// ---------------------------------------------------------------------------
// Node Implementation
// ---------------------------------------------------------------------------

Node::Node(const std::string& node_id,
           const std::string& type,
           const std::string& text_content,
           int page_num,
           const std::vector<float>& bbox,
           Node* parent_node)
    : id(node_id),
      node_type(type),
      content(text_content),
      page_number(page_num),
      bounding_box(bbox),
      parent(parent_node) {
    if (bounding_box.size() < 4) {
        bounding_box.resize(4, 0.0f);
    }
}

Node::~Node() {
    for (Node* child : children) {
        delete child;
    }
    children.clear();
}

std::string Node::to_json_str(int indent_level) const {
    std::string indent(indent_level * 2, ' ');
    std::string child_indent((indent_level + 1) * 2, ' ');
    std::ostringstream ss;

    ss << indent << "{\n";
    ss << child_indent << "\"id\": \"" << escape_json_string(id) << "\",\n";
    ss << child_indent << "\"node_type\": \"" << escape_json_string(node_type) << "\",\n";
    ss << child_indent << "\"page_number\": " << page_number << ",\n";
    ss << child_indent << "\"content\": \"" << escape_json_string(content) << "\",\n";
    
    // Bounding box
    ss << child_indent << "\"bounding_box\": [";
    for (size_t i = 0; i < bounding_box.size(); ++i) {
        ss << bounding_box[i] << (i + 1 < bounding_box.size() ? ", " : "");
    }
    ss << "],\n";

    // Parent ID
    ss << child_indent << "\"parent_id\": " 
       << (parent ? ("\"" + escape_json_string(parent->id) + "\"") : "null") << ",\n";

    // Metadata
    ss << child_indent << "\"metadata\": {";
    size_t count = 0;
    for (const auto& kv : metadata) {
        if (count > 0) ss << ", ";
        ss << "\"" << escape_json_string(kv.first) << "\": \"" << escape_json_string(kv.second) << "\"";
        count++;
    }
    ss << "},\n";

    // Children
    ss << child_indent << "\"children\": [";
    if (!children.empty()) {
        ss << "\n";
        for (size_t i = 0; i < children.size(); ++i) {
            ss << children[i]->to_json_str(indent_level + 2);
            if (i + 1 < children.size()) {
                ss << ",\n";
            } else {
                ss << "\n";
            }
        }
        ss << child_indent;
    }
    ss << "]\n";
    ss << indent << "}";

    return ss.str();
}

// ---------------------------------------------------------------------------
// NArrayTree Implementation
// ---------------------------------------------------------------------------

NArrayTree::NArrayTree(const std::string& doc_id, const std::string& doc_title) {
    root = new Node(doc_id, "DOCUMENT", doc_title, 0, {0.0f, 0.0f, 0.0f, 0.0f}, nullptr);
    node_lookup[doc_id] = root;
}

NArrayTree::~NArrayTree() {
    clear();
}

void NArrayTree::clear() {
    if (root) {
        delete root;
        root = nullptr;
    }
    node_lookup.clear();
    cached_json.clear();
}

Node* NArrayTree::get_node(const std::string& id) const {
    auto it = node_lookup.find(id);
    if (it != node_lookup.end()) {
        return it->second;
    }
    return nullptr;
}

Node* NArrayTree::add_page_node(const std::string& id,
                                const std::string& node_type,
                                const std::string& content,
                                int page_number,
                                const std::vector<float>& bbox,
                                const std::unordered_map<std::string, std::string>& metadata) {
    if (!root) return nullptr;
    if (node_lookup.find(id) != node_lookup.end()) {
        // Node already exists
        return node_lookup[id];
    }

    Node* page_node = new Node(id, node_type, content, page_number, bbox, root);
    page_node->metadata = metadata;
    root->children.push_back(page_node);
    node_lookup[id] = page_node;
    return page_node;
}

Node* NArrayTree::add_child_element(const std::string& parent_id,
                                   const std::string& id,
                                   const std::string& node_type,
                                   const std::string& content,
                                   int page_number,
                                   const std::vector<float>& bbox,
                                   const std::unordered_map<std::string, std::string>& metadata) {
    auto parent_it = node_lookup.find(parent_id);
    if (parent_it == node_lookup.end()) {
        return nullptr; // Parent not found
    }
    if (node_lookup.find(id) != node_lookup.end()) {
        return node_lookup[id]; // Already exists
    }

    Node* parent = parent_it->second;
    Node* child = new Node(id, node_type, content, page_number, bbox, parent);
    child->metadata = metadata;
    parent->children.push_back(child);
    node_lookup[id] = child;
    return child;
}

void NArrayTree::collect_preorder(Node* current, std::vector<Node*>& result) const {
    if (!current) return;
    result.push_back(current);
    for (Node* child : current->children) {
        collect_preorder(child, result);
    }
}

void NArrayTree::collect_bfs(Node* start, std::vector<Node*>& result) const {
    if (!start) return;
    std::queue<Node*> q;
    q.push(start);
    while (!q.empty()) {
        Node* curr = q.front();
        q.pop();
        result.push_back(curr);
        for (Node* child : curr->children) {
            q.push(child);
        }
    }
}

std::vector<Node*> NArrayTree::traverse(const std::string& order) const {
    std::vector<Node*> result;
    if (!root) return result;

    if (order == "BFS") {
        collect_bfs(root, result);
    } else {
        collect_preorder(root, result);
    }
    return result;
}

std::string NArrayTree::export_json() const {
    if (!root) return "{}";
    return root->to_json_str(0);
}

} // namespace amhie

// ---------------------------------------------------------------------------
// Helper for parsing simple JSON key-value pairs (e.g. {"key": "val", ...})
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, std::string> parse_simple_kv_json(const char* json_str) {
    std::unordered_map<std::string, std::string> result;
    if (!json_str || std::strlen(json_str) == 0) return result;

    std::string s(json_str);
    size_t i = 0;
    while (i < s.size()) {
        size_t key_start = s.find('"', i);
        if (key_start == std::string::npos) break;
        size_t key_end = s.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = s.substr(key_start + 1, key_end - key_start - 1);

        size_t colon = s.find(':', key_end + 1);
        if (colon == std::string::npos) break;

        size_t val_start = s.find_first_not_of(" \t\r\n", colon + 1);
        if (val_start == std::string::npos) break;

        std::string val;
        if (s[val_start] == '"') {
            size_t val_end = s.find('"', val_start + 1);
            if (val_end == std::string::npos) break;
            val = s.substr(val_start + 1, val_end - val_start - 1);
            i = val_end + 1;
        } else {
            size_t val_end = s.find_first_of(",}", val_start);
            if (val_end == std::string::npos) val_end = s.size();
            val = s.substr(val_start, val_end - val_start);
            // trim whitespace
            val.erase(val.find_last_not_of(" \t\r\n") + 1);
            i = val_end;
        }
        result[key] = val;
    }
    return result;
}

// ---------------------------------------------------------------------------
// C ABI Interface Implementation
// ---------------------------------------------------------------------------
extern "C" {

void* amhie_tree_create(const char* doc_id, const char* doc_title) {
    std::string id = doc_id ? doc_id : "doc_root";
    std::string title = doc_title ? doc_title : "Document";
    return new amhie::NArrayTree(id, title);
}

void amhie_tree_destroy(void* tree_ptr) {
    if (tree_ptr) {
        delete static_cast<amhie::NArrayTree*>(tree_ptr);
    }
}

int amhie_tree_add_page_node(void* tree_ptr,
                             const char* id,
                             const char* node_type,
                             const char* content,
                             int page_number,
                             float x0, float y0, float x1, float y1,
                             const char* metadata_pairs_json) {
    if (!tree_ptr || !id || !node_type) return 0;
    auto* tree = static_cast<amhie::NArrayTree*>(tree_ptr);
    std::vector<float> bbox = {x0, y0, x1, y1};
    auto metadata = parse_simple_kv_json(metadata_pairs_json);

    amhie::Node* node = tree->add_page_node(id, node_type, content ? content : "", page_number, bbox, metadata);
    return node != nullptr ? 1 : 0;
}

int amhie_tree_add_child_element(void* tree_ptr,
                                 const char* parent_id,
                                 const char* id,
                                 const char* node_type,
                                 const char* content,
                                 int page_number,
                                 float x0, float y0, float x1, float y1,
                                 const char* metadata_pairs_json) {
    if (!tree_ptr || !parent_id || !id || !node_type) return 0;
    auto* tree = static_cast<amhie::NArrayTree*>(tree_ptr);
    std::vector<float> bbox = {x0, y0, x1, y1};
    auto metadata = parse_simple_kv_json(metadata_pairs_json);

    amhie::Node* node = tree->add_child_element(parent_id, id, node_type, content ? content : "", page_number, bbox, metadata);
    return node != nullptr ? 1 : 0;
}

const char* amhie_tree_export_json(void* tree_ptr) {
    if (!tree_ptr) return nullptr;
    auto* tree = static_cast<amhie::NArrayTree*>(tree_ptr);
    std::string json = tree->export_json();
    char* c_str = new char[json.size() + 1];
    std::strcpy(c_str, json.c_str());
    return c_str;
}

void amhie_tree_free_string(const char* str) {
    if (str) {
        delete[] str;
    }
}

int amhie_tree_get_size(void* tree_ptr) {
    if (!tree_ptr) return 0;
    auto* tree = static_cast<amhie::NArrayTree*>(tree_ptr);
    return static_cast<int>(tree->size());
}

} // extern "C"

#ifdef AMHIE_BUILD_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(amhie_tree, m) {
    m.doc() = "AMHIE High-Performance C++ N-ary Structural Document Tree Engine";

    py::class_<amhie::Node>(m, "Node")
        .def_readonly("id", &amhie::Node::id)
        .def_readonly("node_type", &amhie::Node::node_type)
        .def_readonly("content", &amhie::Node::content)
        .def_readonly("page_number", &amhie::Node::page_number)
        .def_readonly("bounding_box", &amhie::Node::bounding_box)
        .def_readonly("metadata", &amhie::Node::metadata)
        .def_readonly("children", &amhie::Node::children)
        .def("to_json", &amhie::Node::to_json_str, py::arg("indent_level") = 0);

    py::class_<amhie::NArrayTree>(m, "NArrayTree")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("doc_id") = "doc_root", py::arg("doc_title") = "Document")
        .def("get_root", &amhie::NArrayTree::get_root, py::return_value_policy::reference)
        .def("size", &amhie::NArrayTree::size)
        .def("get_node", &amhie::NArrayTree::get_node, py::arg("id"), py::return_value_policy::reference)
        .def("add_page_node", &amhie::NArrayTree::add_page_node,
             py::arg("id"), py::arg("node_type"), py::arg("content"),
             py::arg("page_number"), py::arg("bbox") = std::vector<float>{0,0,0,0},
             py::arg("metadata") = std::unordered_map<std::string, std::string>{},
             py::return_value_policy::reference)
        .def("add_child_element", &amhie::NArrayTree::add_child_element,
             py::arg("parent_id"), py::arg("id"), py::arg("node_type"),
             py::arg("content"), py::arg("page_number"),
             py::arg("bbox") = std::vector<float>{0,0,0,0},
             py::arg("metadata") = std::unordered_map<std::string, std::string>{},
             py::return_value_policy::reference)
        .def("traverse", &amhie::NArrayTree::traverse, py::arg("order") = "PREORDER", py::return_value_policy::reference)
        .def("export_json", &amhie::NArrayTree::export_json)
        .def("clear", &amhie::NArrayTree::clear);
}
#endif
