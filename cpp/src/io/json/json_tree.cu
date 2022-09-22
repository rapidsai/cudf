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
#include <hash/hash_allocator.cuh>
#include <hash/helper_functions.cuh>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/detail/hashing.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <cuco/static_map.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <limits>

namespace cudf::io::json {
namespace detail {

// DEBUG print
namespace {
template <typename T>
void print_vec(T const& cpu, std::string const name)
{
  for (auto const& v : cpu)
    printf("%3d,", int(v));
  std::cout << name << std::endl;
}
}  // namespace

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
  nvtxRangePushA("node_categories");
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
  nvtxRangePop();

  nvtxRangePushA("token_levels");
  // Node levels: transform_exclusive_scan, copy_if.
  rmm::device_uvector<size_type> token_levels(num_tokens, stream);
  auto push_pop_it = thrust::make_transform_iterator(
    tokens.begin(), [does_push, does_pop] __device__(PdaTokenT const token) -> size_type {
      return does_push(token) - does_pop(token);
    });
  thrust::exclusive_scan(
    rmm::exec_policy(stream), push_pop_it, push_pop_it + num_tokens, token_levels.begin());
  nvtxRangePop();

  nvtxRangePushA("node_levels");
  rmm::device_uvector<TreeDepthT> node_levels(num_nodes, stream, mr);
  auto node_levels_end = thrust::copy_if(rmm::exec_policy(stream),
                                         token_levels.begin(),
                                         token_levels.begin() + num_tokens,
                                         tokens.begin(),
                                         node_levels.begin(),
                                         is_node);
  CUDF_EXPECTS(node_levels_end - node_levels.begin() == num_nodes, "node level count mismatch");
  nvtxRangePop();

  nvtxRangePushA("node_range");
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
  nvtxRangePop();

  nvtxRangePushA("node_parent_ids");
  // Node parent ids: previous push token_id transform, stable sort, segmented scan with Max,
  // reorder, copy_if. This one is sort of logical stack. But more generalized.
  // TODO: make it own function.
  rmm::device_uvector<size_type> parent_token_ids(num_tokens, stream);
  rmm::device_uvector<size_type> initial_order(num_tokens, stream);
  nvtxRangePushA("seq, tabulate");
  thrust::sequence(rmm::exec_policy(stream), initial_order.begin(), initial_order.end());
  thrust::tabulate(rmm::exec_policy(stream),
                   parent_token_ids.begin(),
                   parent_token_ids.end(),
                   [does_push, tokens_gpu = tokens.begin()] __device__(auto i) -> size_type {
                     return (i > 0) && does_push(tokens_gpu[i - 1]) ? i - 1 : -1;
                     // XXX: How is this algorithm working for JSON lines?
                   });
  nvtxRangePop();

  nvtxRangePushA("sort-level");
  auto out_pid = thrust::make_zip_iterator(parent_token_ids.data(), initial_order.data());
  // Uses radix sort for builtin types.
  thrust::stable_sort_by_key(rmm::exec_policy(stream),
                             token_levels.data(),
                             token_levels.data() + token_levels.size(),
                             out_pid);
  nvtxRangePop();

  nvtxRangePushA("scan-level");
  // SegmentedScan Max.
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                token_levels.data(),
                                token_levels.data() + token_levels.size(),
                                parent_token_ids.data(),
                                parent_token_ids.data(),
                                thrust::equal_to<size_type>{},
                                thrust::maximum<size_type>{});
  // Reusing token_levels memory & use scatter to restore the original order.
  nvtxRangePop();

  nvtxRangePushA("scatter");
  std::swap(token_levels, parent_token_ids);
  auto& sorted_parent_token_ids = token_levels;
  thrust::scatter(rmm::exec_policy(stream),
                  sorted_parent_token_ids.begin(),
                  sorted_parent_token_ids.end(),
                  initial_order.data(),
                  parent_token_ids.data());
  nvtxRangePop();

  nvtxRangePushA("node_ids-scan");
  rmm::device_uvector<size_type> node_ids_gpu(num_tokens, stream);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), is_node_it, is_node_it + num_tokens, node_ids_gpu.begin());
  nvtxRangePop();

  nvtxRangePushA("parent_node_ids-copy_if");
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
  nvtxRangePop();
  nvtxRangePop();  // node_parent_ids
  return {std::move(node_categories),
          std::move(parent_node_ids),
          std::move(node_levels),
          std::move(node_range_begin),
          std::move(node_range_end)};
}

/**
@note
This algorithm assigns a unique column id to each node in the tree.
The row offset is the row index of the node in that column id.
Algorithm:
1. Convert node_category+fieldname to node_type.
  a. Create a hashmap to hash field name and assign unique node id as values.
  b. Convert the node categories to node types.
     Node type is defined as node category enum value if it is not a field node,
     otherwise it is the unique node id assigned by the hashmap (value shifted by #NUM_CATEGORY).
2. Preprocessing: Translate parent node ids after sorting by level.
  a. sort by level
  b. get gather map of sorted indices
  c. translate parent_node_ids to new sorted indices
3. Find level boundaries.
   copy_if index of first unique values of sorted levels.
4. Per-Level Processing: Propagate parent node ids for each level.
  For each level,
    a. gather col_id from previous level results. input=col_id, gather_map is parent_indices.
    b. stable sort by {parent_col_id, node_type}
    c. scan sum of unique {parent_col_id, node_type}
    d. scatter the col_id back to stable node_level order (using scatter_indices)
  Restore original node_id order
5. Generate row_offset.
  a. stable_sort by parent_col_id.
  b. scan_by_key {parent_col_id} (required only on nodes who's parent is list)
  c. propagate to non-list leaves from parent list node by recursion
**/
std::tuple<rmm::device_uvector<NodeIndexT>, rmm::device_uvector<size_type>>
records_orient_tree_traversal(device_span<SymbolT const> d_input,
                              tree_meta_t& d_tree,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // 1. Convert node_category+fieldname to node_type.
  nvtxRangePushA("node_type");
  auto num_nodes                           = d_tree.node_categories.size();
  rmm::device_uvector<size_type> node_type = [&]() {
    using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;
    using hash_map_type =
      cuco::static_map<size_type, size_type, cuda::thread_scope_device, hash_table_allocator_type>;

    constexpr size_type empty_node_index_sentinel = -1;
    hash_map_type key_map{compute_hash_table_size(num_nodes),  // TODO reduce oversubscription
                          cuco::sentinel::empty_key{empty_node_index_sentinel},
                          cuco::sentinel::empty_value{empty_node_index_sentinel},
                          hash_table_allocator_type{default_allocator<char>{}, stream},
                          stream.value()};
    auto d_hasher = [d_input          = d_input.data(),
                     node_range_begin = d_tree.node_range_begin.data(),
                     node_range_end   = d_tree.node_range_end.data()] __device__(auto node_id) {
      auto const field_name = cudf::string_view(
        d_input + node_range_begin[node_id], node_range_end[node_id] - node_range_begin[node_id]);
      return cudf::detail::default_hash<cudf::string_view>{}(field_name);
    };
    auto d_equal = [d_input          = d_input.data(),
                    node_range_begin = d_tree.node_range_begin.data(),
                    node_range_end   = d_tree.node_range_end.data()] __device__(auto node_id1,
                                                                              auto node_id2) {
      auto const field_name1 =
        cudf::string_view(d_input + node_range_begin[node_id1],
                          node_range_end[node_id1] - node_range_begin[node_id1]);
      auto const field_name2 =
        cudf::string_view(d_input + node_range_begin[node_id2],
                          node_range_end[node_id2] - node_range_begin[node_id2]);
      return field_name1 == field_name2;
    };
    auto is_field_name_node = [node_categories = d_tree.node_categories.data()] __device__(
                                auto node_id) { return node_categories[node_id] == node_t::NC_FN; };
    // key-value pairs: uses node_id itself as node_type. (unique node_id for a field name due to
    // hashing)
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, [] __device__(size_type i) { return cuco::make_pair(i, i); });

    key_map.insert_if(iter,
                      iter + num_nodes,
                      thrust::counting_iterator<size_type>(0),  // stencil
                      is_field_name_node,
                      d_hasher,
                      d_equal,
                      stream.value());
    auto get_hash_value = [key_map = key_map.get_device_view(), d_hasher, d_equal] __device__(
                            auto node_id) -> size_type {
      auto it = key_map.find(node_id, d_hasher, d_equal);
      return (it == key_map.end()) ? size_type{0} : it->second.load();
    };
    // convert field nodes to node indices, and other nodes to enum value.
    rmm::device_uvector<size_type> node_type(num_nodes, stream);
    thrust::tabulate(rmm::exec_policy(stream),
                     node_type.begin(),
                     node_type.end(),
                     [node_categories = d_tree.node_categories.data(),
                      is_field_name_node,
                      get_hash_value] __device__(auto node_id) -> size_type {
                       if (is_field_name_node(node_id))
                         return static_cast<size_type>(NUM_NODE_CLASSES) + get_hash_value(node_id);
                       else
                         return static_cast<size_type>(node_categories[node_id]);
                     });
    return node_type;
  }();
  nvtxRangePop();

  // TODO two-level hashing:  one for field names
  // and another for {node-level, node_category} + field hash for the entire path

#ifdef NJP_DEBUG_PRINT
  print_vec(cudf::detail::make_std_vector_async(node_type, stream), "node_type");
#endif
  // 2. Preprocessing: Translate parent node ids after sorting by level.
  //   a. sort by level
  //   b. get gather map of sorted indices
  //   c. translate parent_node_ids to sorted indices

  nvtxRangePushA("parent_indices");
  rmm::device_uvector<size_type> scatter_indices(num_nodes, stream);
  thrust::sequence(rmm::exec_policy(stream), scatter_indices.begin(), scatter_indices.end());
#ifdef NJP_DEBUG_PRINT
  printf("\n");
  print_vec(cudf::detail::make_std_vector_async(scatter_indices, stream), "gpu.node_id");
  print_vec(cudf::detail::make_std_vector_async(d_tree.parent_node_ids, stream),
            "gpu.parent_node_ids");
  print_vec(cudf::detail::make_std_vector_async(node_type, stream), "gpu.node_type");
  print_vec(cudf::detail::make_std_vector_async(d_tree.node_levels, stream), "gpu.node_levels");
#endif
  rmm::device_uvector<NodeIndexT> parent_node_ids(d_tree.parent_node_ids, stream);  // make a copy
  auto out_pid =
    thrust::make_zip_iterator(scatter_indices.data(), parent_node_ids.data(), node_type.data());
  // Uses cub radix sort.
  thrust::stable_sort_by_key(rmm::exec_policy(stream),
                             d_tree.node_levels.data(),
                             d_tree.node_levels.data() + num_nodes,
                             out_pid);
  auto gather_indices = cudf::detail::scatter_to_gather(
    scatter_indices.begin(), scatter_indices.end(), num_nodes, stream);

  rmm::device_uvector<NodeIndexT> parent_indices(num_nodes, stream);
  // gather, except parent sentinels
  thrust::transform(rmm::exec_policy(stream),
                    parent_node_ids.begin(),
                    parent_node_ids.end(),
                    parent_indices.begin(),
                    [gather_indices = gather_indices.data()] __device__(auto parent_node_id) {
                      return (parent_node_id == parent_node_sentinel)
                               ? parent_node_sentinel
                               : gather_indices[parent_node_id];
                    });
  nvtxRangePop();
#ifdef NJP_DEBUG_PRINT
  printf("\n");
  print_vec(cudf::detail::make_std_vector_async(scatter_indices, stream), "gpu.node_id");
  print_vec(cudf::detail::make_std_vector_async(parent_node_ids, stream), "gpu.parent_node_ids");
  print_vec(cudf::detail::make_std_vector_async(node_type, stream), "gpu.node_type");
  print_vec(cudf::detail::make_std_vector_async(d_tree.node_levels, stream), "gpu.node_levels");
  print_vec(cudf::detail::make_std_vector_async(gather_indices, stream), "new_home");
  print_vec(cudf::detail::make_std_vector_async(parent_indices, stream), "parent_indices");
  print_vec(cudf::detail::make_std_vector_async(parent_node_ids, stream),
            "parent_node_ids (restored)");
#endif
  // 3. Find level boundaries.
  nvtxRangePushA("level_boundaries");
  std::vector<size_type> level_boundaries = [&]() {
    // Already node_levels is sorted
    auto max_level = d_tree.node_levels.back_element(stream);
    rmm::device_uvector<size_type> level_boundaries(max_level + 1, stream);
    // TODO try reduce_by_key
    auto level_end =
      thrust::copy_if(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(1),
                      thrust::make_counting_iterator<size_type>(num_nodes + 1),
                      level_boundaries.begin(),
                      [num_nodes, node_levels = d_tree.node_levels.begin()] __device__(auto index) {
                        return index == num_nodes || node_levels[index] != node_levels[index - 1];
                      });
    CUDF_EXPECTS(thrust::distance(level_boundaries.begin(), level_end) == max_level + 1,
                 "num_levels != max_level + 1");
    return cudf::detail::make_std_vector_async(level_boundaries, stream);
  }();
  nvtxRangePop();
  auto num_levels = level_boundaries.size();
#ifdef NJP_DEBUG_PRINT
  print_vec(level_boundaries, "level_boundaries");
  std::cout << "num_levels: " << num_levels << std::endl;
#endif

#ifdef NJP_DEBUG_PRINT
  auto print_level_data = [stream](auto stage_text,
                                   auto level,
                                   auto start,
                                   auto end,
                                   auto const& nodeII,
                                   auto const& parent_node_idx,
                                   auto const& parent_col_id,
                                   auto const& node_type,
                                   auto const& levels,
                                   auto const& col_id) {
    std::cout << level << stage_text << "\n";
    auto cls = std::array<const char*, 6>{"S", "L", "F", "R", "V", "E"};
    for (auto n = start; n < end; n++)
      printf("%3d ", nodeII.element(n, stream));
    printf(" nodeII-%ld\n", level);
    for (auto n = start; n < end; n++)
      printf("%3d ", parent_node_idx.element(n, stream));
    printf(" parent_node_idx-%ld\n", level);
    for (auto n = start; n < end; n++)
      printf("%3d ", parent_col_id.element(n, stream));
    printf(" parent_col_id-%ld\n", level);
    for (auto n = start; n < end; n++) {
      auto nt = node_type.element(n, stream);
      printf("%3s ",
             nt >= NUM_NODE_CLASSES ? std::to_string(nt - NUM_NODE_CLASSES).c_str() : cls[nt]);
    }
    printf(" node_type-%ld\n", level);
    for (auto n = start; n < end; n++)
      printf("%3d ", levels.element(n, stream));
    printf(" levels-%ld\n", level);
    for (auto n = start; n < end; n++)
      printf("%3d ", col_id.element(n, stream));
    printf(" col_id-%ld\n", level);
  };
#define PRINT_LEVEL_DATA(level, text)           \
  print_level_data(text,                        \
                   level,                       \
                   level_boundaries[level - 1], \
                   level_boundaries[level],     \
                   scatter_indices,             \
                   parent_indices,              \
                   parent_col_id,               \
                   node_type,                   \
                   d_tree.node_levels,          \
                   col_id);
#else
#define PRINT_LEVEL_DATA(level, text) ;
#endif

  // 4. Propagate parent node ids for each level.
  // For each level,
  //     a. gather col_id from previous level results. input=col_id, gather_map is parent_indices.
  //     b. stable sort by {parent_col_id, node_type}
  //     c. scan sum of unique {parent_col_id, node_type}
  //     d. scatter the col_id back to stable node_level order (using scatter_indices)
  nvtxRangePushA("pre-level");
  rmm::device_uvector<NodeIndexT> col_id(num_nodes, stream, mr);
  rmm::device_uvector<NodeIndexT> parent_col_id(num_nodes, stream);
  thrust::uninitialized_fill(rmm::exec_policy(stream),
                             parent_col_id.begin(),
                             parent_col_id.end(),
                             0);  // XXX: is this needed?
  // fill with 1, useful for scan later
  // TODO: Could initialize to 0 and scatter 1 to level_boundaries alone
  thrust::uninitialized_fill(rmm::exec_policy(stream), col_id.begin(), col_id.end(), 1);
  // Initialize First level node's node col_id to 0
  thrust::fill(rmm::exec_policy(stream), col_id.begin(), col_id.begin() + level_boundaries[0], 0);
  // Initialize First level node's parent_col_id to parent_node_sentinel sentinel
  thrust::fill(rmm::exec_policy(stream),
               parent_col_id.begin(),
               parent_col_id.begin() + level_boundaries[0],
               parent_node_sentinel);
  nvtxRangePop();
  nvtxRangePushA("level");
  for (decltype(num_levels) level = 1; level < num_levels; level++) {
    PRINT_LEVEL_DATA(level, ".before gather");
    nvtxRangePushA("gather");
    // Gather the each node's parent's column id for the nodes of the current level
    thrust::gather(rmm::exec_policy(stream),
                   parent_indices.data() + level_boundaries[level - 1],
                   parent_indices.data() + level_boundaries[level],
                   col_id.data(),
                   parent_col_id.data() + level_boundaries[level - 1]);
    PRINT_LEVEL_DATA(level, ".after gather");
    nvtxRangePop();
    nvtxRangePushA("stable_sort_by_key");
    thrust::stable_sort_by_key(
      rmm::exec_policy(stream),
      thrust::make_zip_iterator(parent_col_id.begin() + level_boundaries[level - 1],
                                node_type.data() + level_boundaries[level - 1]),
      thrust::make_zip_iterator(parent_col_id.begin() + level_boundaries[level],
                                node_type.data() + level_boundaries[level]),
      thrust::make_zip_iterator(scatter_indices.begin() + level_boundaries[level - 1]));
    PRINT_LEVEL_DATA(level, ".after sort");
    nvtxRangePop();
    nvtxRangePushA("transform");
    auto start_it = thrust::make_zip_iterator(parent_col_id.begin() + level_boundaries[level - 1],
                                              node_type.data() + level_boundaries[level - 1]);
    auto adjacent_pair_it = thrust::make_zip_iterator(start_it - 1, start_it);
    thrust::transform(rmm::exec_policy(stream),
                      adjacent_pair_it + 1,
                      adjacent_pair_it + level_boundaries[level] - level_boundaries[level - 1],
                      col_id.data() + level_boundaries[level - 1] + 1,
                      [] __device__(auto adjacent_pair) -> size_type {
                        auto lhs = thrust::get<0>(adjacent_pair),
                             rhs = thrust::get<1>(adjacent_pair);
                        return lhs != rhs ? 1 : 0;
                      });
    nvtxRangePop();
    nvtxRangePushA("scan");
    // includes previous level last col_id to continue the index.
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           col_id.data() + level_boundaries[level - 1],
                           col_id.data() + level_boundaries[level] + (level != num_levels - 1),
                           // +1 only for not-last-levels, for next level start col_id
                           col_id.data() + level_boundaries[level - 1]);
    PRINT_LEVEL_DATA(level, ".after scan");
    // TODO scatter/gather to restore original order. (but scatter_indices is not zero based here)
    thrust::sort_by_key(
      rmm::exec_policy(stream),
      scatter_indices.begin() + level_boundaries[level - 1],
      scatter_indices.begin() + level_boundaries[level],
      thrust::make_zip_iterator(col_id.begin() + level_boundaries[level - 1],
                                parent_col_id.data() + level_boundaries[level - 1]));
    PRINT_LEVEL_DATA(level, ".after restore order");
    nvtxRangePop();
  }
  nvtxRangePop();
  nvtxRangePushA("restore");
  // restore original order of col_id., and used d_tree members
  // TODO would scatter be faster than radix-sort here for 3 values?
  thrust::sort_by_key(rmm::exec_policy(stream),
                      scatter_indices.begin(),
                      scatter_indices.end(),
                      thrust::make_zip_iterator(
#ifdef NJP_DEBUG_PRINT
                        parent_indices.begin(),  // only needed for debug prints
                        node_type.begin(),
#endif
                        parent_col_id.begin(),
                        col_id.begin(),
                        d_tree.node_levels.begin()));
#ifdef NJP_DEBUG_PRINT
  auto translate_col_id = [](auto col_id) {
    std::unordered_map<int, int> col_id_map;
    std::vector<int> new_col_ids(col_id.size());
    int unique_id = 0;
    for (auto id : col_id) {
      if (col_id_map.count(id) == 0) { col_id_map[id] = unique_id++; }
    }
    for (size_t i = 0; i < col_id.size(); i++) {
      new_col_ids[i] = col_id_map[col_id[i]];
    }
    return new_col_ids;
  };
  print_vec(cudf::detail::make_std_vector_async(scatter_indices, stream), "gpu.node_id");
  print_vec(cudf::detail::make_std_vector_async(parent_indices, stream),
            "gpu.parent_indices");  // once original order is restored, is this required?
  print_vec(cudf::detail::make_std_vector_async(node_type, stream),
            "gpu.node_type");  // is this needed?
  print_vec(cudf::detail::make_std_vector_async(parent_col_id, stream),
            "parent_col_id");                                                // is this needed?
  print_vec(cudf::detail::make_std_vector_async(col_id, stream), "col_id");  // required.
  print_vec(translate_col_id(cudf::detail::make_std_vector_async(col_id, stream)),
            "col_id (translated)");  // is this required? required to be ordered for the next step?
  print_vec(cudf::detail::make_std_vector_async(d_tree.node_levels, stream), "gpu.node_levels");
#endif

  nvtxRangePop();
  nvtxRangePushA("rowoffset");
  // 5. Generate row_offset.
  //   a. stable_sort by parent_col_id.
  //   b. scan_by_key {parent_col_id} (required only on nodes who's parent is list)
  //   c. propagate to non-list leaves from parent list node by recursion
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream), parent_col_id.begin(), parent_col_id.end(), scatter_indices.begin());
  rmm::device_uvector<size_type> row_offsets(num_nodes, stream, mr);
  // TODO is it possible to generate list child_offsets too here?
  thrust::exclusive_scan_by_key(
    rmm::exec_policy(stream),
    parent_col_id.begin(),  // TODO: is there any way to limit this to list parents alone?
    parent_col_id.end(),
    thrust::make_constant_iterator<size_type>(1),
    row_offsets.begin());
#ifdef NJP_DEBUG_PRINT
  print_vec(cudf::detail::make_std_vector_async(parent_col_id, stream), "parent_col_id");
  print_vec(cudf::detail::make_std_vector_async(row_offsets, stream), "row_offsets (generated)");
#endif
  // Using scatter instead of sort.
  thrust::scatter(rmm::exec_policy(stream),
                  row_offsets.begin(),
                  row_offsets.end(),
                  scatter_indices.begin(),
                  parent_col_id.begin());  // reuse parent_col_id as temp storage
  thrust::copy(
    rmm::exec_policy(stream), parent_col_id.begin(), parent_col_id.end(), row_offsets.begin());
  thrust::transform_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(num_nodes),
    row_offsets.begin(),
    [node_categories = d_tree.node_categories.data(),
     parent_node_ids = d_tree.parent_node_ids.begin(),
     row_offsets     = row_offsets.begin()] __device__(size_type node_id) {
      auto parent_node_id = parent_node_ids[node_id];
      while (parent_node_id != parent_node_sentinel and
             node_categories[parent_node_id] != node_t::NC_LIST) {
        node_id        = parent_node_id;
        parent_node_id = parent_node_ids[parent_node_id];
      }
      return row_offsets[node_id];
    },
    [node_categories = d_tree.node_categories.data(),
     parent_node_ids = d_tree.parent_node_ids.begin()] __device__(size_type node_id) {
      auto parent_node_id = parent_node_ids[node_id];
      return parent_node_id != parent_node_sentinel and
             !(node_categories[parent_node_id] == node_t::NC_LIST);
    });
  nvtxRangePop();
#ifdef NJP_DEBUG_PRINT
  print_vec(cudf::detail::make_std_vector_async(row_offsets, stream), "row_offsets (ordered)");
#endif
  return std::tuple{std::move(col_id), std::move(row_offsets)};
}

}  // namespace detail
}  // namespace cudf::io::json
