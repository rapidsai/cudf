#include <cudf/detail/utilities/vector_factories.hpp>

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

struct trie_device_view {
  uint16_t const* layer_offsets;
  char const* tokens;
  uint16_t const* transitions;
  uint8_t const* match_length;

  inline constexpr uint16_t transition(uint16_t idx, char c)
  {
    auto pos = transitions[idx];
    auto end = transitions[idx + 1];
    while (pos < end) {
      if (c == tokens[pos - 1]) { return pos; }
      pos++;
    }

    return transition_init(c);
  }

  inline constexpr uint16_t transition_init(char c)
  {
    auto pos = transitions[0];
    auto end = transitions[1];
    while (pos < end) {
      if (c == tokens[pos - 1]) { return pos; }
      pos++;
    }

    return 0;
  }

  inline constexpr bool is_match(uint16_t idx) { return static_cast<bool>(get_match_length(idx)); }
  inline constexpr uint8_t get_match_length(uint16_t idx) { return match_length[idx]; }
};

struct trie {
  // could compress all of this to 32 bits without major perf reduction:
  // 1) merge is_accepting state in to the most significant bit of the
  // corrosponding transition, and use a mask to access both values. 2) change
  // layer_offsets to uint8_t, max string length would be 253 2^8-3 (two values
  // reserved: empty string, and error state)
 private:
  rmm::device_uvector<uint16_t> _layer_offsets;
  rmm::device_uvector<char> _tokens;
  rmm::device_uvector<uint16_t> _transitions;
  rmm::device_uvector<uint8_t> _match_length;

 public:
  trie(rmm::device_uvector<uint16_t>&& layer_offsets,
       rmm::device_uvector<char>&& tokens,
       rmm::device_uvector<uint16_t>&& transitions,
       rmm::device_uvector<uint8_t>&& match_length)
    : _layer_offsets(std::move(layer_offsets)),
      _tokens(std::move(tokens)),
      _transitions(std::move(transitions)),
      _match_length(std::move(match_length))
  {
  }

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
    std::vector<uint16_t> layer_offsets;
    std::vector<char> tokens;
    std::vector<uint16_t> transitions;
    std::vector<uint8_t> match_length;

    // create the trie tree
    auto root = std::make_unique<trie_builder_node>();
    for (auto& pattern : patterns) {
      root->insert(pattern);
    }

    // flatten
    auto sum = 1;
    layer_offsets.emplace_back(0);
    transitions.emplace_back(sum);
    match_length.emplace_back(root->match_length);

    auto nodes = std::queue<std::unique_ptr<trie_builder_node>>();
    nodes.push(std::move(root));

    while (nodes.size()) {
      layer_offsets.emplace_back(sum);
      auto layer_size = nodes.size();
      for (uint32_t i = 0; i < layer_size; i++) {
        auto node = std::move(nodes.front());
        nodes.pop();
        sum += node->children.size();
        transitions.emplace_back(sum);
        for (auto& item : node->children) {
          match_length.emplace_back(item.second->match_length);
          tokens.emplace_back(item.first);
          nodes.push(std::move(item.second));
        }
      }
    }

    match_length.emplace_back(false);

    return trie{detail::make_device_uvector_async(layer_offsets, stream, mr),
                detail::make_device_uvector_async(tokens, stream, mr),
                detail::make_device_uvector_async(transitions, stream, mr),
                detail::make_device_uvector_async(match_length, stream, mr)};
  }

  trie_device_view view() const
  {
    return trie_device_view{
      _layer_offsets.data(), _tokens.data(), _transitions.data(), _match_length.data()};
  }
};

}  // namespace text
}  // namespace io
}  // namespace cudf
