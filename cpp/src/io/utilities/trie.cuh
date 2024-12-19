/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

/**
 * @brief Serialized trie implementation for C++/CUDA
 * @file trie.cuh
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <optional>

namespace cudf {
namespace detail {
static constexpr char trie_terminating_character = '\n';

/**
 * @brief Node in the serialized trie.
 *
 * A serialized trie is an array of nodes. Each node represents a matching character, except for the
 * last child node, which denotes the end of the children list. Children of a node are stored
 * contiguously. The `children_offset` member is the offset between the node and its first child.
 * Matching is successful if all characters are matched and the final node is the last character of
 * a word (i.e. `is_leaf` is true).
 *
 */
struct serial_trie_node {
  int16_t children_offset{-1};
  char character{trie_terminating_character};
  bool is_leaf{false};
  explicit serial_trie_node(char c, bool leaf = false) noexcept : character(c), is_leaf(leaf) {}
};

using trie          = rmm::device_uvector<serial_trie_node>;
using optional_trie = std::optional<trie>;
using trie_view     = device_span<serial_trie_node const>;

inline trie_view make_trie_view(optional_trie const& t)
{
  if (!t) return {};
  return trie_view{t->data(), t->size()};
}

/**
 * @brief Creates a serialized trie for cache-friendly string search.
 *
 * The resulting trie is a compact array - children array size is equal to the
 * actual number of children nodes, not the size of the alphabet.
 *
 * @param keys Array of strings to insert into the trie
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return A host vector of nodes representing the serialized trie
 */
CUDF_EXPORT trie create_serialized_trie(std::vector<std::string> const& keys,
                                        rmm::cuda_stream_view stream);

/*
 * @brief Searches for a string in a serialized trie.
 *
 * Can be executed on host or device, as long as the data is available
 *
 * @param trie Pointer to the array of nodes that make up the trie
 * @param key Pointer to the start of the string to find
 * @param key_len Length of the string to find
 *
 * @return Boolean value; true if string is found, false otherwise
 */
CUDF_HOST_DEVICE inline bool serialized_trie_contains(device_span<serial_trie_node const> trie,
                                                      device_span<char const> key)
{
  if (trie.empty()) { return false; }
  if (key.empty()) { return trie.front().is_leaf; }
  auto curr_node = trie.begin() + 1;
  for (auto curr_key = key.begin(); curr_key < key.end(); ++curr_key) {
    // Don't jump away from root node
    if (curr_key != key.begin()) { curr_node += curr_node->children_offset; }
    // Search for the next character in the array of children nodes
    // Nodes are sorted - terminate search if the node is larger or equal
    while (curr_node->character != trie_terminating_character && curr_node->character < *curr_key) {
      ++curr_node;
    }
    // Could not find the next character, done with the search
    if (curr_node->character != *curr_key) { return false; }
  }
  // Even if the node is present, return true only if that node is at the end of a word
  return curr_node->is_leaf;
}
}  // namespace detail
}  // namespace cudf
