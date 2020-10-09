/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
 *
 */

#ifndef TRIE_CUH
#define TRIE_CUH

#include <deque>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>

static constexpr char trie_terminating_character = '\n';

struct SerialTrieNode {
  int16_t children_offset{-1};
  char character{trie_terminating_character};
  bool is_leaf{false};
  SerialTrieNode() = default;  // FIXME This is necessary for a Thrust bug on CentOS7 + CUDA10
  explicit SerialTrieNode(char c, bool leaf = false) noexcept : character(c), is_leaf(leaf) {}
};

/**
 * @brief Create a serialized trie for cache-friendly string search
 *
 * The resulting trie is a compact array - children array size is equal to the
 * actual number of children nodes, not the size of the alphabet
 *
 * @param[in] keys Array of strings to insert into the trie
 *
 * @return A host vector of nodes representing the serialized trie
 */
inline thrust::host_vector<SerialTrieNode> createSerializedTrie(
  const std::vector<std::string> &keys)
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
  for (const auto &key : keys) {
    auto *current_node = &tree_trie;

    for (const char character : key) {
      if (current_node->children[character] == nullptr)
        current_node->children[character] = std::make_unique<TreeTrieNode>();

      current_node = current_node->children[character].get();
    }

    current_node->is_end_of_word = true;
  }

  struct IndexedTrieNode {
    TreeTrieNode const *const pnode;
    int16_t const idx;
    IndexedTrieNode(TreeTrieNode const *const node, int16_t index) : pnode(node), idx(index) {}
  };

  // Serialize the tree trie
  std::deque<IndexedTrieNode> to_visit;
  thrust::host_vector<SerialTrieNode> nodes;
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
        nodes.push_back(SerialTrieNode(static_cast<char>(i), node->children[i]->is_end_of_word));
        // Add to the queue, with the index within the new trie
        to_visit.emplace_back(node->children[i].get(), static_cast<uint16_t>(nodes.size()) - 1);

        has_children = true;
      }
    }
    // Only add the terminating character is there any nodes were added
    if (has_children) { nodes.push_back(SerialTrieNode(trie_terminating_character)); }
  }
  return nodes;
}

/*
 * @brief Searches for a string in a serialized trie
 *
 * Can be executed on host or device, as long as the data is available
 *
 * @param[in] trie Pointer to the array of nodes that make up the trie
 * @param[in] key Pointer to the start of the string to find
 * @param[in] key_len Length of the string to find
 *
 * @return Boolean value, true if string is found, false otherwise
 */
__host__ __device__ inline bool serializedTrieContains(const SerialTrieNode *trie,
                                                       const char *key,
                                                       size_t key_len)
{
  if (trie == nullptr) return false;
  int curr_node = 0;
  for (size_t i = 0; i < key_len; ++i) {
    // Don't jump away from root node
    if (i != 0) { curr_node += trie[curr_node].children_offset; }
    // Search for the next character in the array of children nodes
    // Nodes are sorted - terminate search if the node is larger or equal
    while (trie[curr_node].character != trie_terminating_character &&
           trie[curr_node].character < key[i]) {
      ++curr_node;
    }
    // Could not find the next character, done with the search
    if (trie[curr_node].character != key[i]) { return false; }
  }
  // Even if the node is present, return true only if that node is at the end of a word
  return trie[curr_node].is_leaf;
}

#endif  // TRIE_CUH
