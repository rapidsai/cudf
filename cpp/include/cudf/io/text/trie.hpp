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
  uint8_t child_begin;
};

struct trie_path_part {
  uint32_t head;
  uint32_t tail;
};

struct trie_queue {
  static uint32_t const N = 8;
  trie_path_part values[N];
  uint32_t pos;
  uint32_t end;

  inline constexpr uint32_t size() { return end - pos; }

  inline constexpr trie_path_part peek() { return values[pos % N]; }

  inline constexpr trie_path_part dequeue() { return values[pos++ % N]; }

  inline constexpr void enqueue(trie_path_part value)
  {
    if (size() < N) { values[end++ % N] = value; }
  }
};

struct trie_device_view {
  device_span<trie_node const> _nodes;

  template <uint32_t N>
  inline constexpr void transition_init(  //
    char c,
    trie_path_part (&parts)[N],
    uint32_t& pos,
    uint32_t& end)
  {
    for (uint32_t curr = 0; curr < _nodes.size() - 1; curr++) {
      transition_enqueue_all(c, parts, pos, end, curr, curr);
    }
  }

  template <uint32_t N>
  inline constexpr void transition(  //
    char c,
    trie_path_part (&parts)[N],
    uint32_t& pos,
    uint32_t& end)
  {
    auto size = end - pos;
    transition_enqueue_all(c, parts, pos, end, 0, 0);
    for (uint32_t i = 0; i < size; i++) {
      auto partial = parts[pos++ % N];
      transition_enqueue_all(c, parts, pos, end, partial.head, partial.tail);
    }
  }

  template <uint32_t N>
  inline constexpr void transition_enqueue_all(  //
    char c,
    trie_path_part (&parts)[N],
    uint32_t& pos,
    uint32_t& end,
    uint32_t const& head,
    uint32_t const& curr)
  {
    for (uint32_t tail = _nodes[curr].child_begin; tail < _nodes[curr + 1].child_begin; tail++) {
      if (end - pos < N) {              //
        if (_nodes[tail].token == c) {  //
          parts[end++ % N] = {head, tail};
        }
      }
    }
  }

  inline constexpr bool is_match(uint16_t idx) { return static_cast<bool>(get_match_length(idx)); }
  inline constexpr uint8_t get_match_length(uint16_t idx) { return _nodes[idx].match_length; }

  template <uint32_t N>
  inline constexpr uint8_t get_match_length(trie_path_part (&parts)[N],
                                            uint32_t& pos,
                                            uint32_t& end)
  {
    int8_t val = 0;
    for (uint32_t i = pos; i != end; i++) {
      val = max(val, get_match_length(parts[i % N].tail));
    }
    return val;
  }
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
