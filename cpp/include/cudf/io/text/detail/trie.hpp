/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <cudf/io/text/detail/multistate.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudf {
namespace io {
namespace text {
namespace detail {

struct trie_node {
  char token;
  uint8_t match_length;
  uint8_t child_begin;
};

struct trie_device_view {
  device_span<trie_node const> _nodes;

  /**
   * @brief create a multistate which contains all partial path matches for the given token.
   */
  constexpr multistate transition_init(char c)
  {
    auto result = multistate();

    result.enqueue(0, 0);

    for (uint8_t curr = 0; curr < _nodes.size() - 1; curr++) {
      transition_enqueue_all(c, result, curr, curr);
    }
    return result;
  }

  /**
   * @brief create a new multistate by transitioning all states in the multistate by the given token
   *
   * Eliminates any partial matches that cannot transition using the given token.
   *
   * @note always enqueues (0, 0] as the first state of the returned multistate.
   */
  constexpr multistate transition(char c, multistate const& states)
  {
    auto result = multistate();

    result.enqueue(0, 0);

    for (uint8_t i = 0; i < states.size(); i++) {
      transition_enqueue_all(c, result, states.get_head(i), states.get_tail(i));
    }

    return result;
  }

  /**
   * @brief returns true if the given index is associated with a matching state.
   */
  constexpr bool is_match(uint16_t idx) { return static_cast<bool>(get_match_length(idx)); }

  /**
   * @brief returns the match length if the given index is associated with a matching state,
   * otherwise zero.
   */
  constexpr uint8_t get_match_length(uint16_t idx) { return _nodes[idx].match_length; }

 private:
  constexpr void transition_enqueue_all(  //
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
};

/**
 * @brief A flat trie contained in device memory.
 */
struct trie {
 private:
  cudf::size_type _max_duplicate_tokens;
  rmm::device_uvector<trie_node> _nodes;

  trie(cudf::size_type max_duplicate_tokens, rmm::device_uvector<trie_node>&& nodes)
    : _max_duplicate_tokens(max_duplicate_tokens), _nodes(std::move(nodes))
  {
  }

  /**
   * @brief Used to build a hierarchical trie which can then be flattened.
   */
  struct trie_builder_node {
    uint8_t match_length;
    std::unordered_map<char, std::unique_ptr<trie_builder_node>> children;

    /**
     * @brief Insert the string in to the trie tree, growing the trie as necessary
     */
    void insert(std::string s) { insert(s.c_str(), s.size(), 0); }

   private:
    trie_builder_node& insert(char const* s, uint16_t size, uint8_t depth)
    {
      if (size == 0) {
        match_length = depth;
        return *this;
      }

      if (children[*s] == nullptr) { children[*s] = std::make_unique<trie_builder_node>(); }

      return children[*s]->insert(s + 1, size - 1, depth + 1);
    }
  };

 public:
  /**
   * @brief Gets the number of nodes contained in this trie.
   */
  [[nodiscard]] cudf::size_type size() const { return _nodes.size(); }

  /**
   * @brief A pessimistic count of duplicate tokens in the trie. Used to determine the maximum
   * possible stack size required to compute matches of this trie in parallel.
   */
  [[nodiscard]] cudf::size_type max_duplicate_tokens() const { return _max_duplicate_tokens; }

  /**
   * @brief Create a trie which represents the given pattern.
   *
   * @param pattern The pattern to store in the trie
   * @param stream The stream to use for allocation and copy
   * @param mr Memory resource to use for the device memory allocation
   * @return The trie.
   */
  static trie create(std::string const& pattern,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr)

  {
    return create(std::vector<std::string>{pattern}, stream, mr);
  }

  /**
   * @brief Create a trie which represents the given pattern.
   *
   * @param pattern The patterns to store in the trie
   * @param stream The stream to use for allocation and copy
   * @param mr Memory resource to use for the device memory allocation
   * @return The trie.
   */
  static trie create(std::vector<std::string> const& patterns,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr)
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
    auto token_counts = std::unordered_map<cudf::size_type, int32_t>();

    for (uint32_t i = 0; i < tokens.size(); i++) {
      trie_nodes.emplace_back(trie_node{tokens[i], match_length[i], transitions[i]});
      token_counts[tokens[i]]++;
    }

    auto most_common_token =
      std::max_element(token_counts.begin(), token_counts.end(), [](auto const& a, auto const& b) {
        return a.second < b.second;
      });

    auto max_duplicate_tokens = most_common_token->second;

    return trie{max_duplicate_tokens,
                cudf::detail::make_device_uvector_sync(trie_nodes, stream, mr)};
  }

  [[nodiscard]] trie_device_view view() const { return trie_device_view{_nodes}; }
};

}  // namespace detail
}  // namespace text
}  // namespace io
}  // namespace cudf
