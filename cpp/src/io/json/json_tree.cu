/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "nested_json.hpp"

#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/detail/nvtx/ranges.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf::io::json {
namespace detail {

// The node that a token represents
struct token_to_node {
  __device__ auto operator()(PdaTokenT const token) -> NodeT
  {
    switch (token) {
      case token_t::StructBegin: return NC_STRUCT;
      case token_t::ListBegin: return NC_LIST;
      case token_t::StringBegin: return NC_STR;
      case token_t::ValueBegin: return NC_VAL;
      case token_t::FieldNameBegin: return NC_FN;
      default: return NC_ERR;
    };
  }
};

// Convert token indices to node range for each valid node.
struct node_ranges {
  device_span<PdaTokenT const> tokens;
  device_span<SymbolOffsetT const> token_indices;
  bool include_quote_char;
  __device__ auto operator()(size_type i) -> thrust::tuple<SymbolOffsetT, SymbolOffsetT>
  {
    // Whether a token expects to be followed by its respective end-of-* token partner
    auto is_begin_of_section = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin:
        case token_t::ValueBegin:
        case token_t::FieldNameBegin: return true;
        default: return false;
      };
    };
    // The end-of-* partner token for a given beginning-of-* token
    auto end_of_partner = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin: return token_t::StringEnd;
        case token_t::ValueBegin: return token_t::ValueEnd;
        case token_t::FieldNameBegin: return token_t::FieldNameEnd;
        default: return token_t::ErrorBegin;
      };
    };
    // Includes quote char for end-of-string token or Skips the quote char for
    // beginning-of-field-name token
    auto get_token_index = [include_quote_char = include_quote_char] __device__(
                             PdaTokenT const token, SymbolOffsetT const token_index) {
      constexpr SymbolOffsetT quote_char_size = 1;
      switch (token) {
        // Strip off quote char included for StringBegin
        case token_t::StringBegin: return token_index + (include_quote_char ? 0 : quote_char_size);
        // Strip off or Include trailing quote char for string values for StringEnd
        case token_t::StringEnd: return token_index + (include_quote_char ? quote_char_size : 0);
        // Strip off quote char included for FieldNameBegin
        case token_t::FieldNameBegin: return token_index + quote_char_size;
        default: return token_index;
      };
    };
    PdaTokenT const token = tokens[i];
    // The section from the original JSON input that this token demarcates
    SymbolOffsetT range_begin = get_token_index(token, token_indices[i]);
    SymbolOffsetT range_end   = range_begin + 1;  // non-leaf, non-field nodes ignore this value.
    if (is_begin_of_section(token)) {
      if ((i + 1) < tokens.size() && end_of_partner(token) == tokens[i + 1]) {
        // Update the range_end for this pair of tokens
        range_end = get_token_index(tokens[i + 1], token_indices[i + 1]);
      }
    }
    return thrust::make_tuple(range_begin, range_end);
  }
};

// Generates a tree representation of the given tokens, token_indices.
tree_meta_t get_tree_representation(device_span<PdaTokenT const> tokens,
                                    device_span<SymbolOffsetT const> token_indices,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // Whether a token does represent a node in the tree representation
  auto is_node = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return true;
      default: return false;
    };
  };

  // Whether the token pops from the parent node stack
  auto does_pop = [] __device__(PdaTokenT const token) {
    switch (token) {
      case token_t::StructMemberEnd:
      case token_t::StructEnd:
      case token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto does_push = [] __device__(PdaTokenT const token) {
    switch (token) {
      case token_t::FieldNameBegin:
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  auto num_tokens = tokens.size();
  auto is_node_it = thrust::make_transform_iterator(
    tokens.begin(),
    [is_node] __device__(auto t) -> size_type { return static_cast<size_type>(is_node(t)); });
  auto num_nodes = thrust::count_if(
    rmm::exec_policy(stream), tokens.begin(), tokens.begin() + num_tokens, is_node);

  // Node categories: copy_if with transform.
  rmm::device_uvector<NodeT> node_categories(num_nodes, stream, mr);
  auto node_categories_it =
    thrust::make_transform_output_iterator(node_categories.begin(), token_to_node{});
  auto node_categories_end = thrust::copy_if(rmm::exec_policy(stream),
                                             tokens.begin(),
                                             tokens.begin() + num_tokens,
                                             node_categories_it,
                                             is_node);
  CUDF_EXPECTS(node_categories_end - node_categories_it == num_nodes,
               "node category count mismatch");

  // Node levels: transform_exclusive_scan, copy_if.
  rmm::device_uvector<size_type> token_levels(num_tokens, stream);
  auto push_pop_it = thrust::make_transform_iterator(
    tokens.begin(), [does_push, does_pop] __device__(PdaTokenT const token) -> size_type {
      return does_push(token) - does_pop(token);
    });
  thrust::exclusive_scan(
    rmm::exec_policy(stream), push_pop_it, push_pop_it + num_tokens, token_levels.begin());

  rmm::device_uvector<TreeDepthT> node_levels(num_nodes, stream, mr);
  auto node_levels_end = thrust::copy_if(rmm::exec_policy(stream),
                                         token_levels.begin(),
                                         token_levels.begin() + num_tokens,
                                         tokens.begin(),
                                         node_levels.begin(),
                                         is_node);
  CUDF_EXPECTS(node_levels_end - node_levels.begin() == num_nodes, "node level count mismatch");

  // Node ranges: copy_if with transform.
  rmm::device_uvector<SymbolOffsetT> node_range_begin(num_nodes, stream, mr);
  rmm::device_uvector<SymbolOffsetT> node_range_end(num_nodes, stream, mr);
  auto node_range_tuple_it =
    thrust::make_zip_iterator(node_range_begin.begin(), node_range_end.begin());
  // Whether the tokenizer stage should keep quote characters for string values
  // If the tokenizer keeps the quote characters, they may be stripped during type casting
  constexpr bool include_quote_char = true;
  auto node_range_out_it            = thrust::make_transform_output_iterator(
    node_range_tuple_it, node_ranges{tokens, token_indices, include_quote_char});

  auto node_range_out_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(0) + num_tokens,
                    node_range_out_it,
                    [is_node, tokens_gpu = tokens.begin()] __device__(size_type i) -> bool {
                      return is_node(tokens_gpu[i]);
                    });
  CUDF_EXPECTS(node_range_out_end - node_range_out_it == num_nodes, "node range count mismatch");

  // Node parent ids: previous push token_id transform, stable sort, segmented scan with Max,
  // reorder, copy_if. This one is sort of logical stack. But more generalized.
  // TODO: make it own function.
  rmm::device_uvector<size_type> parent_token_ids(num_tokens, stream);
  rmm::device_uvector<size_type> initial_order(num_tokens, stream);
  thrust::sequence(rmm::exec_policy(stream), initial_order.begin(), initial_order.end());
  thrust::tabulate(rmm::exec_policy(stream),
                   parent_token_ids.begin(),
                   parent_token_ids.end(),
                   [does_push, tokens_gpu = tokens.begin()] __device__(auto i) -> size_type {
                     return (i > 0) && does_push(tokens_gpu[i - 1]) ? i - 1 : -1;
                   });
  auto out_pid = thrust::make_zip_iterator(parent_token_ids.data(), initial_order.data());
  // Uses radix sort for builtin types.
  thrust::stable_sort_by_key(rmm::exec_policy(stream),
                             token_levels.data(),
                             token_levels.data() + token_levels.size(),
                             out_pid);
  // SegmentedScan Max.
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                token_levels.data(),
                                token_levels.data() + token_levels.size(),
                                parent_token_ids.data(),
                                parent_token_ids.data(),  // size_type{-1},
                                thrust::equal_to<size_type>{},
                                thrust::maximum<size_type>{});
  // Reusing token_levels memory & use scatter to restore the original order.
  std::swap(token_levels, parent_token_ids);
  auto& sorted_parent_token_ids = token_levels;
  thrust::scatter(rmm::exec_policy(stream),
                  sorted_parent_token_ids.begin(),
                  sorted_parent_token_ids.end(),
                  initial_order.data(),
                  parent_token_ids.data());

  rmm::device_uvector<size_type> node_ids_gpu(num_tokens, stream);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), is_node_it, is_node_it + num_tokens, node_ids_gpu.begin());
  rmm::device_uvector<NodeIndexT> parent_node_ids(num_nodes, stream, mr);
  auto parent_node_ids_it = thrust::make_transform_iterator(
    parent_token_ids.begin(),
    [node_ids_gpu = node_ids_gpu.begin()] __device__(size_type const pid) -> NodeIndexT {
      return pid < 0 ? pid : node_ids_gpu[pid];
    });
  auto parent_node_ids_end = thrust::copy_if(rmm::exec_policy(stream),
                                             parent_node_ids_it,
                                             parent_node_ids_it + parent_token_ids.size(),
                                             tokens.begin(),
                                             parent_node_ids.begin(),
                                             is_node);
  CUDF_EXPECTS(parent_node_ids_end - parent_node_ids.begin() == num_nodes,
               "parent node id gather mismatch");
  return {std::move(node_categories),
          std::move(parent_node_ids),
          std::move(node_levels),
          std::move(node_range_begin),
          std::move(node_range_end)};
}

}  // namespace detail
}  // namespace cudf::io::json
