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

/**
 * @brief Serialized trie implementation for C++/CUDA
 * @file trie.cu
 */

#include "trie.cuh"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda_runtime.h>

#include <deque>
#include <string>
#include <vector>

namespace cudf {
namespace detail {

rmm::device_uvector<serial_trie_node> create_serialized_trie(const std::vector<std::string>& keys,
                                                             rmm::cuda_stream_view stream)
{
  static constexpr int alphabet_size = std::numeric_limits<char>::max() + 1;
  struct TreeTrieNode {
    using TrieNodePtr                 = std::unique_ptr<TreeTrieNode>;
    std::vector<TrieNodePtr> children = std::vector<TrieNodePtr>(alphabet_size);
    bool is_end_of_word               = false;
  };

  // Construct a tree-structured trie
  // The trie takes a lot of memory, but the lookup is fast:
  // allows direct addressing of children nodes
  TreeTrieNode tree_trie;
  for (const auto& key : keys) {
    auto* current_node = &tree_trie;

    for (const char character : key) {
      if (current_node->children[character] == nullptr)
        current_node->children[character] = std::make_unique<TreeTrieNode>();

      current_node = current_node->children[character].get();
    }

    current_node->is_end_of_word = true;
  }

  struct IndexedTrieNode {
    TreeTrieNode const* const pnode;
    int16_t const idx;
    IndexedTrieNode(TreeTrieNode const* const node, int16_t index) : pnode(node), idx(index) {}
  };

  // Serialize the tree trie
  std::deque<IndexedTrieNode> to_visit;
  std::vector<serial_trie_node> nodes;

  // If the Tree trie matches empty strings, the root node is marked as 'end of word'.
  // The first node in the serialized trie is also used to match empty strings, so we're
  // initializing it using the `is_end_of_word` value from the root node.
  nodes.push_back(serial_trie_node(trie_terminating_character, tree_trie.is_end_of_word));

  // Add root node to queue. this node is not included to the serialized trie
  to_visit.emplace_back(&tree_trie, -1);
  while (!to_visit.empty()) {
    const auto node_and_idx = to_visit.front();
    const auto node         = node_and_idx.pnode;
    const auto idx          = node_and_idx.idx;
    to_visit.pop_front();

    bool has_children = false;
    for (size_t i = 0; i < node->children.size(); ++i) {
      if (node->children[i] != nullptr) {
        // Update the children offset of the parent node, unless at the root
        if (idx >= 0 && nodes[idx].children_offset < 0) {
          nodes[idx].children_offset = static_cast<uint16_t>(nodes.size() - idx);
        }
        // Add node to the trie
        nodes.push_back(serial_trie_node(static_cast<char>(i), node->children[i]->is_end_of_word));
        // Add to the queue, with the index within the new trie
        to_visit.emplace_back(node->children[i].get(), static_cast<uint16_t>(nodes.size()) - 1);

        has_children = true;
      }
    }
    // Only add the terminating character if any nodes were added
    if (has_children) { nodes.push_back(serial_trie_node(trie_terminating_character)); }
  }
  return cudf::detail::make_device_uvector_sync(nodes, stream);
}

}  // namespace detail
}  // namespace cudf
