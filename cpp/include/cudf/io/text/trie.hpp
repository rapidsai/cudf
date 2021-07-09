#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <queue>
#include <string>
#include <vector>

namespace {

struct trie_builder_node {
  bool is_accepting;
  std::unordered_map<char, std::unique_ptr<trie_builder_node>> children;

  void insert(std::string s) { insert(s.c_str(), s.size()); }

  trie_builder_node& insert(char const* s, uint16_t size)
  {
    if (size == 0) {
      is_accepting = true;
      return *this;
    }

    if (children[*s] == nullptr) { children[*s] = std::make_unique<trie_builder_node>(); }

    return children[*s]->insert(s + 1, size - 1);
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
  bool const* accepting;

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

  inline constexpr bool is_match(uint16_t idx) { return accepting[idx]; }
};

struct trie {
  // could compress all of this to 32 bits without major perf reduction:
  // 1) merge accepting state in to the most significant bit of the
  // corrosponding transition, and use a mask to access both values. 2) change
  // layer_offsets to uint8_t, max string length would be 253 2^8-3 (two values
  // reserved: empty string, and error state)
 private:
  rmm::device_uvector<uint16_t> _layer_offsets;
  rmm::device_uvector<char> _tokens;
  rmm::device_uvector<uint16_t> _transitions;
  rmm::device_uvector<bool> _accepting;

 public:
  trie(rmm::device_uvector<uint16_t>&& layer_offsets,
       rmm::device_uvector<char>&& tokens,
       rmm::device_uvector<uint16_t>&& transitions,
       rmm::device_uvector<bool>&& accepting)
    : _layer_offsets(std::move(layer_offsets)),
      _tokens(std::move(tokens)),
      _transitions(std::move(transitions)),
      _accepting(std::move(accepting))
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
    std::vector<uint8_t> accepting;

    // create the trie tree
    auto root = std::make_unique<trie_builder_node>();
    for (auto& pattern : patterns) { root->insert(pattern); }

    // flatten
    auto sum = 1;
    layer_offsets.emplace_back(0);
    transitions.emplace_back(sum);
    accepting.emplace_back(root->is_accepting);

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
          accepting.emplace_back(item.second->is_accepting);
          tokens.emplace_back(item.first);
          nodes.push(std::move(item.second));
        }
      }
    }

    accepting.emplace_back(false);

    // allocate device memory

    auto device_layer_offsets = rmm::device_uvector<uint16_t>(layer_offsets.size(), stream, mr);
    auto device_tokens        = rmm::device_uvector<char>(tokens.size(), stream, mr);
    auto device_transitions   = rmm::device_uvector<uint16_t>(transitions.size(), stream, mr);
    auto device_accepting     = rmm::device_uvector<bool>(accepting.size(), stream, mr);

    // copy host buffers to device

    RMM_CUDA_TRY(cudaMemcpyAsync(device_layer_offsets.data(),
                                 layer_offsets.data(),
                                 layer_offsets.size() * sizeof(uint16_t),
                                 cudaMemcpyDefault,
                                 stream.value()));

    RMM_CUDA_TRY(cudaMemcpyAsync(device_tokens.data(),
                                 tokens.data(),
                                 tokens.size() * sizeof(char),
                                 cudaMemcpyDefault,
                                 stream.value()));

    RMM_CUDA_TRY(cudaMemcpyAsync(device_transitions.data(),
                                 transitions.data(),
                                 transitions.size() * sizeof(uint16_t),
                                 cudaMemcpyDefault,
                                 stream.value()));

    RMM_CUDA_TRY(cudaMemcpyAsync(device_accepting.data(),
                                 accepting.data(),
                                 accepting.size() * sizeof(bool),
                                 cudaMemcpyDefault,
                                 stream.value()));

    // create owning container

    return trie{std::move(device_layer_offsets),
                std::move(device_tokens),
                std::move(device_transitions),
                std::move(device_accepting)};
  }

  trie_device_view view() const
  {
    return trie_device_view{
      _layer_offsets.data(), _tokens.data(), _transitions.data(), _accepting.data()};
  }
};

}  // namespace text
}  // namespace io
}  // namespace cudf
