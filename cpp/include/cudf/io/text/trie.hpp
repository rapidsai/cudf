#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <queue>
#include <string>
#include <vector>

namespace {

struct trie_builder_node {
  uint8_t match_length;
  std::unordered_map<char, std::unique_ptr<trie_builder_node>> children;

  void insert(std::string s) { insert(s.c_str(), s.size()); }

  trie_builder_node& insert(char const* s, uint16_t size, uint8_t depth = 0)
  {
    if (size == 0) {
      match_length = depth;
      return *this;
    }

    if (children[*s] == nullptr) { children[*s] = std::make_unique<trie_builder_node>(); }

    return children[*s]->insert(s + 1, size - 1, depth + 1);
  }
};

}  // namespace

namespace cudf {
namespace io {
namespace text {

struct trie_node {
  char token;
  uint8_t match_length;
  uint8_t transitions_begin;
};

struct trie_device_view {
  device_span<trie_node const> _nodes;

  inline constexpr uint16_t transition(uint16_t idx, char c)
  {
    auto pos = _nodes[idx].transitions_begin;
    auto end = _nodes[idx + 1].transitions_begin;
    while (pos < end) {
      if (c == _nodes[pos].token) { return pos; }
      pos++;
    }

    return transition_init(c);
  }

  inline constexpr uint16_t transition_init(char c)
  {
    auto pos = _nodes[0].transitions_begin;
    auto end = _nodes[1].transitions_begin;
    while (pos < end) {
      if (c == _nodes[pos].token) { return pos; }
      pos++;
    }

    return 0;
  }

  inline constexpr bool is_match(uint16_t idx) { return static_cast<bool>(get_match_length(idx)); }
  inline constexpr uint8_t get_match_length(uint16_t idx) { return _nodes[idx].match_length; }
};

struct trie {
  // could compress all of this to 32 bits without major perf reduction:
  // 1) merge is_accepting state in to the most significant bit of the
  // corrosponding transition, and use a mask to access both values. 2) change
  // layer_offsets to uint8_t, max string length would be 253 2^8-3 (two values
  // reserved: empty string, and error state)
 private:
  rmm::device_uvector<trie_node> _nodes;

 public:
  trie(rmm::device_uvector<trie_node>&& nodes) : _nodes(std::move(nodes)) {}

  static trie create(std::string const& pattern,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())

  {
    return create(std::vector<std::string>{pattern}, stream, mr);
  }

  static trie create(std::vector<std::string> const& patterns,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
  {
    std::vector<char> tokens;
    std::vector<uint8_t> transitions;
    std::vector<uint8_t> match_length;

    // create the trie tree
    auto root = std::make_unique<trie_builder_node>();
    for (auto& pattern : patterns) {
      root->insert(pattern);
    }

    // flatten
    auto sum = 1;
    transitions.emplace_back(sum);
    match_length.emplace_back(root->match_length);

    auto builder_nodes = std::queue<std::unique_ptr<trie_builder_node>>();
    builder_nodes.push(std::move(root));

    tokens.emplace_back(0);

    while (builder_nodes.size()) {
      auto layer_size = builder_nodes.size();
      for (uint32_t i = 0; i < layer_size; i++) {
        auto node = std::move(builder_nodes.front());
        builder_nodes.pop();
        sum += node->children.size();
        transitions.emplace_back(sum);
        for (auto& item : node->children) {
          match_length.emplace_back(item.second->match_length);
          tokens.emplace_back(item.first);
          builder_nodes.push(std::move(item.second));
        }
      }
    }

    tokens.emplace_back(0);

    match_length.emplace_back(0);

    std::vector<trie_node> trie_nodes;

    for (uint32_t i = 0; i < tokens.size(); i++) {
      trie_nodes.emplace_back(trie_node{tokens[i], match_length[i], transitions[i]});
    }

    return trie{detail::make_device_uvector_async(trie_nodes, stream, mr)};
  }

  trie_device_view view() const { return trie_device_view{_nodes}; }
};

}  // namespace text
}  // namespace io
}  // namespace cudf
