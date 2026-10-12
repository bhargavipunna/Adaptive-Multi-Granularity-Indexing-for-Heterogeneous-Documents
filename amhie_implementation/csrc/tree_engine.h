#ifndef TREE_ENGINE_H
#define TREE_ENGINE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>

namespace amhie {

/**
 * @brief Represents a single structural node in the AMHIE document hierarchy.
 * Node types: DOCUMENT, PAGE_COARSE, PAGE_COMPLEX, TEXT, TABLE, IMAGE
 */
struct Node {
    std::string id;
    std::string node_type;
    std::string content;
    int page_number;
    std::vector<float> bounding_box; // [x0, y0, x1, y1]
    std::unordered_map<std::string, std::string> metadata;
    std::vector<Node*> children;
    Node* parent;

    Node(const std::string& node_id,
         const std::string& type,
         const std::string& text_content,
         int page_num,
         const std::vector<float>& bbox = {0.0f, 0.0f, 0.0f, 0.0f},
         Node* parent_node = nullptr);

    ~Node();

    // Prevent copy
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    std::string to_json_str(int indent_level = 0) const;
};

/**
 * @brief High-performance N-ary Tree data structure for managing 
 * multi-granularity document layouts.
 */
class NArrayTree {
private:
    Node* root;
    std::unordered_map<std::string, Node*> node_lookup;
    mutable std::string cached_json;

    void collect_preorder(Node* current, std::vector<Node*>& result) const;
    void collect_bfs(Node* start, std::vector<Node*>& result) const;

public:
    NArrayTree(const std::string& doc_id = "doc_root", const std::string& doc_title = "Document");
    ~NArrayTree();

    Node* get_root() const { return root; }
    size_t size() const { return node_lookup.size(); }
    Node* get_node(const std::string& id) const;

    Node* add_page_node(const std::string& id,
                        const std::string& node_type,
                        const std::string& content,
                        int page_number,
                        const std::vector<float>& bbox = {0.0f, 0.0f, 0.0f, 0.0f},
                        const std::unordered_map<std::string, std::string>& metadata = {});

    Node* add_child_element(const std::string& parent_id,
                            const std::string& id,
                            const std::string& node_type,
                            const std::string& content,
                            int page_number,
                            const std::vector<float>& bbox = {0.0f, 0.0f, 0.0f, 0.0f},
                            const std::unordered_map<std::string, std::string>& metadata = {});

    std::vector<Node*> traverse(const std::string& order = "PREORDER") const;
    std::string export_json() const;
    void clear();
};

} // namespace amhie

// ---------------------------------------------------------------------------
// C ABI Interface for dynamic loading via Python ctypes
// ---------------------------------------------------------------------------
extern "C" {
    void* amhie_tree_create(const char* doc_id, const char* doc_title);
    void amhie_tree_destroy(void* tree_ptr);
    int amhie_tree_add_page_node(void* tree_ptr,
                                 const char* id,
                                 const char* node_type,
                                 const char* content,
                                 int page_number,
                                 float x0, float y0, float x1, float y1,
                                 const char* metadata_pairs_json);
    int amhie_tree_add_child_element(void* tree_ptr,
                                     const char* parent_id,
                                     const char* id,
                                     const char* node_type,
                                     const char* content,
                                     int page_number,
                                     float x0, float y0, float x1, float y1,
                                     const char* metadata_pairs_json);
    const char* amhie_tree_export_json(void* tree_ptr);
    void amhie_tree_free_string(const char* str);
    int amhie_tree_get_size(void* tree_ptr);
}

#endif // TREE_ENGINE_H
