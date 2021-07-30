/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/text/multistate.hpp>
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

struct trie_device_view {
  device_span<trie_node const> _nodes;

  inline constexpr multistate transition_init(char c)
  {
    auto result = multistate();

    result.enqueue(0, 0);

    for (uint8_t curr = 0; curr < _nodes.size() - 1; curr++) {
      transition_enqueue_all(c, result, curr, curr);
    }
    return result;
  }

  inline constexpr multistate transition(char c, multistate const& states)
  {
    auto result = multistate();

    result.enqueue(0, 0);

    for (uint8_t i = 0; i < states.size(); i++) {
      transition_enqueue_all(c, result, states.get_head(i), states.get_tail(i));
    }

    return result;
  }

  inline constexpr void transition_enqueue_all(  //
    char c,
    multistate& states,
    uint8_t head,
    uint8_t curr)
  {
    for (uint32_t tail = _nodes[curr].child_begin; tail < _nodes[curr + 1].child_begin; tail++) {
      if (_nodes[tail].token == c) {  //
        states.enqueue(head, tail);
      }
    }
  }

  inline constexpr bool is_match(uint16_t idx) { return static_cast<bool>(get_match_length(idx)); }
  inline constexpr uint8_t get_match_length(uint16_t idx) { return _nodes[idx].match_length; }

  template <uint32_t N>
  inline constexpr uint8_t get_match_length(multistate const& states)
  {
    int8_t val = 0;
    for (uint8_t i = 0; i < states.size(); i++) {
      auto match_length = get_match_length(states.get_tail(i));
      if (match_length > val) { val = match_length; }
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
