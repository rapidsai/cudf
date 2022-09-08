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
#include <cudf/utilities/span.hpp>

#include <cuco/static_map.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
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
template <typename T>
void print_vec(T const& cpu, std::string const name)
{
  for (auto const& v : cpu)
    printf("%3d,", int(v));
  std::cout << name << std::endl;
}

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
template <typename T1, typename T2, typename T3>
struct node_ranges {
  T1 tokens;
  T2 token_indices;
  T3 num_tokens;
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
    SymbolOffsetT range_end   = range_begin + 1;
    if (is_begin_of_section(token)) {
      if ((i + 1) < num_tokens && end_of_partner(token) == tokens[i + 1]) {
        // Update the range_end for this pair of tokens
        range_end = token_indices[i + 1];
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
  auto is_node = [] __device__(PdaTokenT const token) -> size_type {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return 1;
      default: return 0;
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
  auto is_node_it = thrust::make_transform_iterator(tokens.begin(), is_node);
  auto num_nodes  = thrust::reduce(rmm::exec_policy(stream), is_node_it, is_node_it + num_tokens);

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
      return does_push(token) ? 1 : (does_pop(token) ? -1 : 0);
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
  using node_ranges_t =
    node_ranges<decltype(tokens.begin()), decltype(token_indices.begin()), decltype(num_tokens)>;
  auto node_range_out_it = thrust::make_transform_output_iterator(
    node_range_tuple_it,
    node_ranges_t{tokens.begin(), token_indices.begin(), num_tokens, include_quote_char});

  auto node_range_out_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(0) + num_tokens,
                    node_range_out_it,
                    [is_node, tokens_gpu = tokens.begin()] __device__(size_type i) -> bool {
                      PdaTokenT const token = tokens_gpu[i];
                      return is_node(token);
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
                     if (i == 0)
                       return -1;
                     else
                       return does_push(tokens_gpu[i - 1]) ? i - 1 : -1;
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
  // FIXME: Avoid sorting again by scatter + extra memory, or permutation iterator for
  // parent_token_ids. Tradeoff?
  thrust::sort_by_key(rmm::exec_policy(stream),
                      initial_order.data(),
                      initial_order.data() + initial_order.size(),
                      parent_token_ids.data());
  // thrust::scatter(rmm::exec_policy(stream),
  //                parent_token_ids.begin(),
  //                parent_token_ids.end(),
  //                initial_order.data(),
  //                parent_token_ids.begin()); //same location not allowed in scatter
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

// JSON tree traversal for record orient. (list of structs)
// returns col_id of each node, and row_offset(TODO)
void records_orient_tree_traversal(device_span<SymbolT const> d_input,
                                   tree_meta_t& d_tree,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // GPU version
  // 3. convert node_category+fieldname to node_type!
  using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;
  using hash_map_type =
    cuco::static_map<size_type, size_type, cuda::thread_scope_device, hash_table_allocator_type>;

  constexpr size_type empty_node_index_sentinel = std::numeric_limits<size_type>::max();
  auto num_nodes                                = d_tree.node_categories.size();
  hash_map_type key_map{compute_hash_table_size(num_nodes),
                        cuco::sentinel::empty_key{empty_node_index_sentinel},
                        cuco::sentinel::empty_value{empty_node_index_sentinel},
                        hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value()};
  auto d_hasher = [d_input          = d_input.data(),
                   node_range_begin = d_tree.node_range_begin.data(),
                   node_range_end   = d_tree.node_range_end.data()] __device__(auto node_id) {
    // TODO if node_category is hashed, then no need for transform later. for field only, use string
    // hash too.
    auto field_name = cudf::string_view(d_input + node_range_begin[node_id],
                                        node_range_end[node_id] - node_range_begin[node_id]);
    return cudf::detail::default_hash<cudf::string_view>{}(field_name);
  };
  auto d_equal = [d_input          = d_input.data(),
                  node_range_begin = d_tree.node_range_begin.data(),
                  node_range_end   = d_tree.node_range_end.data()] __device__(auto node_id1,
                                                                            auto node_id2) {
    // TODO if node_category is used, then no need for transform later.
    auto field_name1 = cudf::string_view(d_input + node_range_begin[node_id1],
                                         node_range_end[node_id1] - node_range_begin[node_id1]);
    auto field_name2 = cudf::string_view(d_input + node_range_begin[node_id2],
                                         node_range_end[node_id2] - node_range_begin[node_id2]);
    return field_name1 == field_name2;
  };
  auto is_field_node = [node_categories = d_tree.node_categories.data()] __device__(auto node_id) {
    return node_categories[node_id] == node_t::NC_FN;
  };
  // key-value pairs: uses node_id itself as node_type. (unique node_id for a field name due to
  // hashing)
  auto iter = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(size_type i) { return cuco::make_pair(i, i); });

  key_map.insert_if(iter,
                    iter + num_nodes,
                    thrust::counting_iterator<size_type>(0),  // stencil
                    is_field_node,
                    d_hasher,
                    d_equal,
                    stream.value());
  auto get_hash_value =
    [key_map = key_map.get_device_view(), d_hasher, d_equal] __device__(auto node_id) -> size_type {
    auto it = key_map.find(node_id, d_hasher, d_equal);
    return (it == key_map.end()) ? size_type{0} : it->second.load();
  };
  // convert field nodes to node indices, and other nodes to enum value.
  rmm::device_uvector<size_type> node_type(num_nodes, stream);
  thrust::tabulate(rmm::exec_policy(stream),
                   node_type.begin(),
                   node_type.end(),
                   [node_categories = d_tree.node_categories.data(),
                    is_field_node,
                    get_hash_value] __device__(auto node_id) -> size_type {
                     if (is_field_node(node_id))
                       return static_cast<size_type>(NUM_NODE_CLASSES) + get_hash_value(node_id);
                     else
                       return static_cast<size_type>(node_categories[node_id]);
                   });
  print_vec(cudf::detail::make_std_vector_async(node_type, stream), "node_type");
  // 1. Preprocessing: Translate parent node ids after sorting by level.
  //   a. sort by level
  //   b. get gather map of sorted indices
  //   c. translate parent_node_ids to sorted indices
  rmm::device_uvector<size_type> scatter_indices(num_nodes, stream);
  thrust::sequence(rmm::exec_policy(stream), scatter_indices.begin(), scatter_indices.end());
  printf("\n");
  print_vec(cudf::detail::make_std_vector_async(scatter_indices, stream), "gpu.node_id");
  print_vec(cudf::detail::make_std_vector_async(d_tree.parent_node_ids, stream),
            "gpu.parent_node_ids");
  print_vec(cudf::detail::make_std_vector_async(node_type, stream), "gpu.node_type");
  print_vec(cudf::detail::make_std_vector_async(d_tree.node_levels, stream), "gpu.node_levels");
  auto out_pid = thrust::make_zip_iterator(scatter_indices.data(),
                                           //  d_tree.node_levels.data(),
                                           d_tree.parent_node_ids.data(),
                                           node_type.data());
  //  d_tree.node_categories.data());
  // TODO: use cub radix sort.
  thrust::stable_sort_by_key(rmm::exec_policy(stream),
                             d_tree.node_levels.data(),
                             d_tree.node_levels.data() + num_nodes,
                             out_pid);
  auto gather_indices = cudf::detail::scatter_to_gather(
    scatter_indices.begin(), scatter_indices.end(), num_nodes, stream);

  rmm::device_uvector<NodeIndexT> parent_indices(num_nodes, stream);
  *thrust::device_pointer_cast(parent_indices.data()) = -1;
  thrust::gather(rmm::exec_policy(stream),
                 d_tree.parent_node_ids.begin() + 1,  // first node's parent is -1
                 d_tree.parent_node_ids.end(),
                 gather_indices.begin(),
                 parent_indices.begin() + 1);
  printf("\n");
  print_vec(cudf::detail::make_std_vector_async(scatter_indices, stream), "gpu.node_id");
  print_vec(cudf::detail::make_std_vector_async(d_tree.parent_node_ids, stream),
            "gpu.parent_node_ids");
  print_vec(cudf::detail::make_std_vector_async(node_type, stream), "gpu.node_type");
  print_vec(cudf::detail::make_std_vector_async(d_tree.node_levels, stream), "gpu.node_levels");
  print_vec(cudf::detail::make_std_vector_async(gather_indices, stream), "new_home");
  print_vec(cudf::detail::make_std_vector_async(parent_indices, stream), "parent_indices");
  // XXX: restore parent_node_ids order using scatter. (check if this order is right?)
  rmm::device_uvector<NodeIndexT> parent_node_ids(num_nodes, stream);  // Used later for row_offsets
  thrust::scatter(rmm::exec_policy(stream),
                  d_tree.parent_node_ids.begin(),
                  d_tree.parent_node_ids.end(),
                  scatter_indices.begin(),
                  parent_node_ids.begin());
  print_vec(cudf::detail::make_std_vector_async(parent_node_ids, stream),
            "parent_node_ids (restored)");
  // 2. Find level boundaries.
  hostdevice_vector<size_type> level_boundaries(num_nodes + 1, stream);
  auto level_end = thrust::copy_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(1),
    thrust::make_counting_iterator<size_type>(num_nodes + 1),
    level_boundaries.d_begin(),
    [num_nodes, node_levels = d_tree.node_levels.begin()] __device__(auto index) {
      return index == 0 || index == num_nodes || node_levels[index] != node_levels[index - 1];
    });
  level_boundaries.device_to_host(stream, true);
  print_vec(level_boundaries, "level_boundaries");
  auto num_levels = level_end - level_boundaries.d_begin();
  std::cout << "num_levels: " << num_levels << std::endl;
  // level_boundaries[num_levels] = num_nodes;

  auto print_level_data = [stream](auto level,
                                   auto start,
                                   auto end,
                                   auto const& nodeII,
                                   auto const& parent_node_idx,
                                   auto const& parent_col_id,
                                   auto const& node_type,
                                   auto const& levels,
                                   auto const& col_id) {
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

  // 4. Propagate parent node ids for each level.
  // For each level,
  //     a. gather col_id from previous level results. input=col_id, gather_map is parent_indices.
  //     b. sort by {col_id, type}
  //     c. scan sum of unique {parent_col_id, type}
  // cross check outputs.
  // Calculate row offsets too.
  rmm::device_uvector<size_type> col_id(num_nodes, stream);
  rmm::device_uvector<size_type> parent_col_id(num_nodes, stream);
  thrust::uninitialized_fill(rmm::exec_policy(stream),
                             parent_col_id.begin(),
                             parent_col_id.end(),
                             0);  // XXX: is this needed?
  thrust::uninitialized_fill(rmm::exec_policy(stream), col_id.begin(), col_id.end(), 0);  ///
  thrust::device_pointer_cast(parent_col_id.data())[0] = -1;
  for (decltype(num_levels) level = 1; level < num_levels; level++) {
    // std::cout << level << ".before gather\n";
    thrust::gather(rmm::exec_policy(stream),
                   parent_indices.data() +
                     level_boundaries[level - 1],  // FIXME: might be wrong. might be a bug here.
                   parent_indices.data() + level_boundaries[level],
                   col_id.data(),  // + level_boundaries[level - 1],
                   parent_col_id.data() + level_boundaries[level - 1]);
    // std::cout << level << ".after gather\n";
    // print_level_data(level,
    //                  level_boundaries[level - 1],
    //                  level_boundaries[level],
    //                  scatter_indices,
    //                  parent_indices,
    //                  parent_col_id,
    //                  node_type,
    //                  d_tree.node_levels,
    //                  col_id);
    // std::cout << level << ".before sort\n";
    // TODO probably sort_by_key value should be a gather/scatter index to restore original order.
    thrust::stable_sort_by_key(
      rmm::exec_policy(stream),
      thrust::make_zip_iterator(parent_col_id.begin() + level_boundaries[level - 1],
                                node_type.data() + level_boundaries[level - 1]),
      thrust::make_zip_iterator(parent_col_id.begin() + level_boundaries[level],
                                node_type.data() + level_boundaries[level]),
      thrust::make_zip_iterator(
        scatter_indices.begin() +
        level_boundaries[level - 1]  //, // is this required?
                                     //  gather_indices.begin() + level_boundaries[level - 1],
                                     //  parent_indices.begin() + level_boundaries[level - 1]
        ));
    // std::cout << level << ".after sort\n";
    // print_level_data(level,
    //                  level_boundaries[level - 1],
    //                  level_boundaries[level],
    //                  scatter_indices,
    //                  parent_indices,
    //                  parent_col_id,
    //                  node_type,
    //                  d_tree.node_levels,
    //                  col_id);
    auto start_it = thrust::make_zip_iterator(parent_col_id.begin() + level_boundaries[level - 1],
                                              node_type.data() + level_boundaries[level - 1]);
    auto adjacent_pair_it = thrust::make_zip_iterator(start_it - 1, start_it);
    // std::cout << level << ".before transform\n";
    thrust::transform(rmm::exec_policy(stream),
                      adjacent_pair_it,
                      adjacent_pair_it + level_boundaries[level] - level_boundaries[level - 1],
                      col_id.data() + level_boundaries[level - 1],
                      [] __device__(auto adjacent_pair) -> size_type {
                        auto lhs = thrust::get<0>(adjacent_pair),
                             rhs = thrust::get<1>(adjacent_pair);
                        return lhs != rhs ? 1 : 0;
                      });
    // std::cout << level << ".before scan\n";
    // // includes previous level last col_id to continue the index.
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           col_id.data() + level_boundaries[level - 1] - 1,
                           col_id.data() + level_boundaries[level],
                           col_id.data() + level_boundaries[level - 1] - 1);
    // // print node_id, parent_node_idx, parent_col_id, node_type, level.
    // std::cout << level << ".after scan\n";
    // print_level_data(level,
    //                  level_boundaries[level - 1],
    //                  level_boundaries[level],
    //                  scatter_indices,
    //                  parent_indices,
    //                  parent_col_id,
    //                  node_type,
    //                  d_tree.node_levels,
    //                  col_id);
    // TODO scatter/gather to restore original order. (scatter will be faster.)
    thrust::sort_by_key(
      rmm::exec_policy(stream),
      scatter_indices.begin() + level_boundaries[level - 1],
      scatter_indices.begin() + level_boundaries[level],
      thrust::make_zip_iterator(col_id.begin() + level_boundaries[level - 1],
                                parent_col_id.data() + level_boundaries[level - 1]));
    // print_level_data(level,
    //                  level_boundaries[level - 1],
    //                  level_boundaries[level],
    //                  scatter_indices,
    //                  parent_indices,
    //                  parent_col_id,
    //                  node_type,
    //                  d_tree.node_levels,
    //                  col_id);
  }
  // FIXME: to make parent_col_id of last level correct, do we need a gather here?
  thrust::gather(rmm::exec_policy(stream),
                 parent_indices.begin() +
                   level_boundaries[num_levels - 1],  // FIXME: might be wrong. might be a bug here.
                 parent_indices.end(),
                 col_id.data(),  // + level_boundaries[level - 1],
                 parent_col_id.data() + level_boundaries[num_levels - 1]);
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
  // restore original order of col_id.
  // TODO can we do this with scatter instead of sort?
  thrust::sort_by_key(rmm::exec_policy(stream),
                      scatter_indices.begin(),
                      scatter_indices.end(),
                      thrust::make_zip_iterator(parent_indices.begin(),
                                                node_type.begin(),
                                                parent_col_id.begin(),
                                                col_id.begin(),
                                                d_tree.node_levels.begin()));
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
  // auto sorted_cpu_col_id = [&]() {
  //     auto sc = cudf::detail::make_std_vector_async(scatter_indices, stream);
  //     std::vector<size_type> sorted_cpu_col_id(sc.size());
  //     for(decltype(sc.size()) i=0; i<sc.size(); i++) {
  //       sorted_cpu_col_id[sc[i]] = node_ids[i];
  //     }
  //     return sorted_cpu_col_id;
  // }();
  // print_vec(sorted_cpu_col_id, "cpu.node_id (sorted)");

  // auto sc = cudf::detail::make_std_vector_async(scatter_indices, stream);
  // for(int i=0; i< int(cpu_tree.node_range_begin.size()); i++) {
  //   printf("%3s ", std::string(input.data() + cpu_tree.node_range_begin[sc[i]],
  //   cpu_tree.node_range_end[sc[i]] - cpu_tree.node_range_begin[sc[i]]).c_str());
  // }
  // printf(" (JSON)\n");

  // 5. Generate row_offset.
  // stable_sort by parent_col_id.
  // scan_by_key on nodes who's parent is list on col_id.
  // propagate to leaves! how?
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream), parent_col_id.begin(), parent_col_id.end(), scatter_indices.begin());
  rmm::device_uvector<size_type> row_offsets(num_nodes, stream);
  // TODO is it possible to generate list child_offsets too here?
  thrust::exclusive_scan_by_key(
    rmm::exec_policy(stream),
    parent_col_id.begin(),  // TODO: is there any way to limit this to list parents alone?
    parent_col_id.end(),
    thrust::make_constant_iterator<size_type>(1),
    row_offsets.begin());
  print_vec(cudf::detail::make_std_vector_async(parent_col_id, stream), "parent_col_id");
  print_vec(cudf::detail::make_std_vector_async(row_offsets, stream), "row_offsets (generated)");
  thrust::sort_by_key(rmm::exec_policy(stream),
                      scatter_indices.begin(),
                      scatter_indices.end(),
                      thrust::make_zip_iterator(parent_col_id.begin(), row_offsets.begin()));
  thrust::transform_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(num_nodes),
    row_offsets.begin(),
    [node_categories = d_tree.node_categories.data(),
     parent_node_ids = parent_node_ids.begin(),
     row_offsets     = row_offsets.begin()] __device__(size_type node_id) {
      auto parent_node_id = parent_node_ids[node_id];
      while (node_categories[parent_node_id] != node_t::NC_LIST &&
             parent_node_id != -1) {  // TODO replace -1 with sentinel
        node_id        = parent_node_id;
        parent_node_id = parent_node_ids[parent_node_id];
      }
      return row_offsets[node_id];
    },
    [node_categories = d_tree.node_categories.data(),
     parent_node_ids = parent_node_ids.begin()] __device__(size_type node_id) {
      auto parent_node_id = parent_node_ids[node_id];
      return parent_node_id != -1 and
             !(node_categories[parent_node_id] ==
               node_t::NC_LIST);  // Parent is not a list, or sentinel/root (might be different
                                  // condition for JSON_lines)
    });
  print_vec(cudf::detail::make_std_vector_async(row_offsets, stream), "row_offsets (generated)");
  // For now: simple while loop for each thread to retrieve parents row_offset until a node's parent
  // is list node. thrust::transform(rmm::exec_policy(stream), //parent node_id, node_category.
  // problem with using parent_col_id is that it may not be null literal. scan operation is fine but
  // how? propagate to leaves in parallel? does it have to be done level by level? need not be
  // because there may be lists in between. revert back the order and simple scan_max is enough?
  // won't work. regardless of order, a simple scan of op(a,b): return if b==0? a: b; will work.
  // (need to be associative.)

  // TODO return col_id, row_offset of each node.
}

}  // namespace detail
}  // namespace cudf::io::json
