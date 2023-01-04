/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cub/device/device_radix_sort.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <limits>

namespace cudf::io::json {
namespace detail {
namespace {

/**
 * @brief Utility for calling thrust::copy_if
 *
 * Workaround for thrust::copy_if bug (https://github.com/NVIDIA/thrust/issues/1302)
 * where it cannot iterate over int-max values `distance(first,last) > int-max`
 * This calls thrust::copy_if in 2B chunks instead.
 */
template <typename InputIterator,
          typename StencilIterator,
          typename OutputIterator,
          typename Predicate>
OutputIterator thrust_copy_if(rmm::exec_policy policy,
                              InputIterator first,
                              InputIterator last,
                              StencilIterator stencil,
                              OutputIterator result,
                              Predicate pred)
{
  auto const copy_size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                                  static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto itr = first;
  while (itr != last) {
    auto const copy_end =
      static_cast<std::size_t>(std::distance(itr, last)) <= copy_size ? last : itr + copy_size;
    result = thrust::copy_if(policy, itr, copy_end, stencil, result, pred);
    stencil += std::distance(itr, copy_end);
    itr = copy_end;
  }
  return result;
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator thrust_copy_if(rmm::exec_policy policy,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              Predicate pred)
{
  return thrust_copy_if(policy, first, last, first, result, pred);
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
      case token_t::ValueBegin:
        return NC_STR;  // NC_VAL;
      // NV_VAL is removed because type inference and
      // reduce_to_column_tree category collapsing takes care of this.
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
    auto const is_begin_of_section = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin:
        case token_t::ValueBegin:
        case token_t::FieldNameBegin: return true;
        default: return false;
      };
    };
    // The end-of-* partner token for a given beginning-of-* token
    auto const end_of_partner = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin: return token_t::StringEnd;
        case token_t::ValueBegin: return token_t::ValueEnd;
        case token_t::FieldNameBegin: return token_t::FieldNameEnd;
        default: return token_t::ErrorBegin;
      };
    };
    // Includes quote char for end-of-string token or Skips the quote char for
    // beginning-of-field-name token
    auto const get_token_index = [include_quote_char = include_quote_char] __device__(
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

/**
 * @brief Returns stable sorted keys and its sorted order
 *
 * Uses cub stable radix sort. The order is internally generated, hence it saves a copy and memory.
 * Since the key and order is returned, using double buffer helps to avoid extra copy to user
 * provided output iterator.
 *
 * @tparam IndexType sorted order type
 * @tparam KeyType key type
 * @param keys keys to sort
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Sorted keys and indices producing that sorted order
 */
template <typename IndexType = size_t, typename KeyType>
std::pair<rmm::device_uvector<KeyType>, rmm::device_uvector<IndexType>> stable_sorted_key_order(
  cudf::device_span<KeyType const> keys, rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // Determine temporary device storage requirements
  rmm::device_uvector<KeyType> keys_buffer1(keys.size(), stream);
  rmm::device_uvector<KeyType> keys_buffer2(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer1(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer2(keys.size(), stream);
  cub::DoubleBuffer<IndexType> order_buffer(order_buffer1.data(), order_buffer2.data());
  cub::DoubleBuffer<KeyType> keys_buffer(keys_buffer1.data(), keys_buffer2.data());
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
    nullptr, temp_storage_bytes, keys_buffer, order_buffer, keys.size());
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  thrust::copy(rmm::exec_policy(stream), keys.begin(), keys.end(), keys_buffer1.begin());
  thrust::sequence(rmm::exec_policy(stream), order_buffer1.begin(), order_buffer1.end());

  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_buffer,
                                  order_buffer,
                                  keys.size(),
                                  0,
                                  sizeof(KeyType) * 8,
                                  stream.value());

  return std::pair{keys_buffer.Current() == keys_buffer1.data() ? std::move(keys_buffer1)
                                                                : std::move(keys_buffer2),
                   order_buffer.Current() == order_buffer1.data() ? std::move(order_buffer1)
                                                                  : std::move(order_buffer2)};
}

/**
 * @brief Propagate parent node to siblings from first sibling.
 *
 * @param node_levels Node levels of each node
 * @param parent_node_ids parent node ids initialized for first child of each push node,
 *                       and other siblings are initialized to -1.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void propagate_parent_to_siblings(cudf::device_span<TreeDepthT const> node_levels,
                                  cudf::device_span<NodeIndexT> parent_node_ids,
                                  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  auto [sorted_node_levels, sorted_order] = stable_sorted_key_order<size_type>(node_levels, stream);
  // instead of gather, using permutation_iterator, which is ~17% faster

  thrust::inclusive_scan_by_key(
    rmm::exec_policy(stream),
    sorted_node_levels.begin(),
    sorted_node_levels.end(),
    thrust::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
    thrust::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
    thrust::equal_to<TreeDepthT>{},
    thrust::maximum<NodeIndexT>{});
}

// Generates a tree representation of the given tokens, token_indices.
tree_meta_t get_tree_representation(device_span<PdaTokenT const> tokens,
                                    device_span<SymbolOffsetT const> token_indices,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // Whether a token does represent a node in the tree representation
  auto const is_node = [] __device__(PdaTokenT const token) -> bool {
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
  auto const does_pop = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::StructMemberEnd:
      case token_t::StructEnd:
      case token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto const does_push = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::FieldNameBegin:
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  // Look for ErrorBegin and report the point of error.
  if (auto const error_count =
        thrust::count(rmm::exec_policy(stream), tokens.begin(), tokens.end(), token_t::ErrorBegin);
      error_count > 0) {
    auto const error_location =
      thrust::find(rmm::exec_policy(stream), tokens.begin(), tokens.end(), token_t::ErrorBegin);
    SymbolOffsetT error_index;
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(&error_index,
                      token_indices.data() + thrust::distance(tokens.begin(), error_location),
                      sizeof(SymbolOffsetT),
                      cudaMemcpyDefault,
                      stream.value()));
    stream.synchronize();
    CUDF_FAIL("JSON Parser encountered an invalid format at location " +
              std::to_string(error_index));
  }

  auto const num_tokens = tokens.size();
  auto const num_nodes =
    thrust::count_if(rmm::exec_policy(stream), tokens.begin(), tokens.end(), is_node);

  // Node levels: transform_exclusive_scan, copy_if.
  rmm::device_uvector<TreeDepthT> node_levels(num_nodes, stream, mr);
  {
    rmm::device_uvector<TreeDepthT> token_levels(num_tokens, stream);
    auto const push_pop_it = thrust::make_transform_iterator(
      tokens.begin(), [does_push, does_pop] __device__(PdaTokenT const token) -> size_type {
        return does_push(token) - does_pop(token);
      });
    thrust::exclusive_scan(
      rmm::exec_policy(stream), push_pop_it, push_pop_it + num_tokens, token_levels.begin());

    auto const node_levels_end = thrust_copy_if(rmm::exec_policy(stream),
                                                token_levels.begin(),
                                                token_levels.end(),
                                                tokens.begin(),
                                                node_levels.begin(),
                                                is_node);
    CUDF_EXPECTS(thrust::distance(node_levels.begin(), node_levels_end) == num_nodes,
                 "node level count mismatch");
  }

  // Node parent ids:
  // previous push node_id transform, stable sort by level, segmented scan with Max, reorder.
  rmm::device_uvector<NodeIndexT> parent_node_ids(num_nodes, stream, mr);
  // This block of code is generalized logical stack algorithm. TODO: make this a separate function.
  {
    rmm::device_uvector<NodeIndexT> node_token_ids(num_nodes, stream);
    thrust_copy_if(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<NodeIndexT>(0),
                   thrust::make_counting_iterator<NodeIndexT>(0) + num_tokens,
                   tokens.begin(),
                   node_token_ids.begin(),
                   is_node);

    // previous push node_id
    // if previous node is a push, then i-1
    // if previous node is FE, then i-2 (returns FB's index)
    // if previous node is SMB and its previous node is a push, then i-2
    // eg. `{ SMB FB FE VB VE SME` -> `{` index as FB's parent.
    // else -1
    auto const first_childs_parent_token_id = [tokens_gpu =
                                                 tokens.begin()] __device__(auto i) -> NodeIndexT {
      if (i <= 0) { return -1; }
      if (tokens_gpu[i - 1] == token_t::StructBegin or tokens_gpu[i - 1] == token_t::ListBegin) {
        return i - 1;
      } else if (tokens_gpu[i - 1] == token_t::FieldNameEnd) {
        return i - 2;
      } else if (tokens_gpu[i - 1] == token_t::StructMemberBegin and
                 (tokens_gpu[i - 2] == token_t::StructBegin ||
                  tokens_gpu[i - 2] == token_t::ListBegin)) {
        return i - 2;
      } else {
        return -1;
      }
    };

    thrust::transform(
      rmm::exec_policy(stream),
      node_token_ids.begin(),
      node_token_ids.end(),
      parent_node_ids.begin(),
      [node_ids_gpu = node_token_ids.begin(), num_nodes, first_childs_parent_token_id] __device__(
        NodeIndexT const tid) -> NodeIndexT {
        auto const pid = first_childs_parent_token_id(tid);
        return pid < 0
                 ? parent_node_sentinel
                 : thrust::lower_bound(thrust::seq, node_ids_gpu, node_ids_gpu + num_nodes, pid) -
                     node_ids_gpu;
        // parent_node_sentinel is -1, useful for segmented max operation below
      });
  }
  // Propagate parent node to siblings from first sibling - inplace.
  propagate_parent_to_siblings(
    cudf::device_span<TreeDepthT const>{node_levels.data(), node_levels.size()},
    parent_node_ids,
    stream);

  // Node categories: copy_if with transform.
  rmm::device_uvector<NodeT> node_categories(num_nodes, stream, mr);
  auto const node_categories_it =
    thrust::make_transform_output_iterator(node_categories.begin(), token_to_node{});
  auto const node_categories_end = thrust_copy_if(
    rmm::exec_policy(stream), tokens.begin(), tokens.end(), node_categories_it, is_node);
  CUDF_EXPECTS(node_categories_end - node_categories_it == num_nodes,
               "node category count mismatch");

  // Node ranges: copy_if with transform.
  rmm::device_uvector<SymbolOffsetT> node_range_begin(num_nodes, stream, mr);
  rmm::device_uvector<SymbolOffsetT> node_range_end(num_nodes, stream, mr);
  auto const node_range_tuple_it =
    thrust::make_zip_iterator(node_range_begin.begin(), node_range_end.begin());
  // Whether the tokenizer stage should keep quote characters for string values
  // If the tokenizer keeps the quote characters, they may be stripped during type casting
  constexpr bool include_quote_char = true;
  auto const node_range_out_it      = thrust::make_transform_output_iterator(
    node_range_tuple_it, node_ranges{tokens, token_indices, include_quote_char});

  auto const node_range_out_end =
    thrust_copy_if(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<size_type>(0),
                   thrust::make_counting_iterator<size_type>(0) + num_tokens,
                   node_range_out_it,
                   [is_node, tokens_gpu = tokens.begin()] __device__(size_type i) -> bool {
                     return is_node(tokens_gpu[i]);
                   });
  CUDF_EXPECTS(node_range_out_end - node_range_out_it == num_nodes, "node range count mismatch");

  return {std::move(node_categories),
          std::move(parent_node_ids),
          std::move(node_levels),
          std::move(node_range_begin),
          std::move(node_range_end)};
}

/**
 * @brief Generates unique node_type id for each node.
 * Field nodes with the same name are assigned the same node_type id.
 * List, Struct, and String nodes are assigned their category values as node_type ids.
 *
 * All inputs and outputs are in node_id order.
 * @param d_input JSON string in device memory
 * @param d_tree Tree representation of the JSON
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Vector of node_type ids
 */
rmm::device_uvector<size_type> hash_node_type_with_field_name(device_span<SymbolT const> d_input,
                                                              tree_meta_t const& d_tree,
                                                              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;
  using hash_map_type =
    cuco::static_map<size_type, size_type, cuda::thread_scope_device, hash_table_allocator_type>;

  auto const num_nodes  = d_tree.node_categories.size();
  auto const num_fields = thrust::count(rmm::exec_policy(stream),
                                        d_tree.node_categories.begin(),
                                        d_tree.node_categories.end(),
                                        node_t::NC_FN);

  constexpr size_type empty_node_index_sentinel = -1;
  hash_map_type key_map{compute_hash_table_size(num_fields, 40),  // 40% occupancy in hash map
                        cuco::sentinel::empty_key{empty_node_index_sentinel},
                        cuco::sentinel::empty_value{empty_node_index_sentinel},
                        hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value()};
  auto const d_hasher = [d_input          = d_input.data(),
                         node_range_begin = d_tree.node_range_begin.data(),
                         node_range_end   = d_tree.node_range_end.data()] __device__(auto node_id) {
    auto const field_name = cudf::string_view(d_input + node_range_begin[node_id],
                                              node_range_end[node_id] - node_range_begin[node_id]);
    return cudf::detail::default_hash<cudf::string_view>{}(field_name);
  };
  auto const d_equal = [d_input          = d_input.data(),
                        node_range_begin = d_tree.node_range_begin.data(),
                        node_range_end   = d_tree.node_range_end.data()] __device__(auto node_id1,
                                                                                  auto node_id2) {
    auto const field_name1 = cudf::string_view(
      d_input + node_range_begin[node_id1], node_range_end[node_id1] - node_range_begin[node_id1]);
    auto const field_name2 = cudf::string_view(
      d_input + node_range_begin[node_id2], node_range_end[node_id2] - node_range_begin[node_id2]);
    return field_name1 == field_name2;
  };
  // key-value pairs: uses node_id itself as node_type. (unique node_id for a field name due to
  // hashing)
  auto const iter = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(size_type i) { return cuco::make_pair(i, i); });

  auto const is_field_name_node = [node_categories =
                                     d_tree.node_categories.data()] __device__(auto node_id) {
    return node_categories[node_id] == node_t::NC_FN;
  };
  key_map.insert_if(iter,
                    iter + num_nodes,
                    thrust::counting_iterator<size_type>(0),  // stencil
                    is_field_name_node,
                    d_hasher,
                    d_equal,
                    stream.value());

  auto const get_hash_value =
    [key_map = key_map.get_device_view(), d_hasher, d_equal] __device__(auto node_id) -> size_type {
    auto const it = key_map.find(node_id, d_hasher, d_equal);
    return (it == key_map.end()) ? size_type{0} : it->second.load(cuda::std::memory_order_relaxed);
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
}

// Two level hashing algorithm
// 1. Convert node_category+fieldname to node_type. (passed as argument)
//   a. Create a hashmap to hash field name and assign unique node id as values.
//   b. Convert the node categories to node types.
//      Node type is defined as node category enum value if it is not a field node,
//      otherwise it is the unique node id assigned by the hashmap (value shifted by #NUM_CATEGORY).
// 2. Set operation on entire path of each node
//   a. Create a hash map with hash of {node_level, node_type} of its node and the entire parent
//      until root.
//   b. While creating hashmap, transform node id to unique node ids that are inserted into the
//      hash map. This mimics set operation with hash map. This unique node ids are set ids.
//   c. Return this converted set ids, which are the hash map keys/values, and unique set ids.
std::pair<rmm::device_uvector<size_type>, rmm::device_uvector<size_type>> hash_node_path(
  device_span<TreeDepthT const> node_levels,
  device_span<size_type const> node_type,
  device_span<NodeIndexT const> parent_node_ids,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto const num_nodes = parent_node_ids.size();
  rmm::device_uvector<size_type> col_id(num_nodes, stream, mr);

  using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;
  using hash_map_type =
    cuco::static_map<size_type, size_type, cuda::thread_scope_device, hash_table_allocator_type>;

  constexpr size_type empty_node_index_sentinel = -1;
  hash_map_type key_map{compute_hash_table_size(num_nodes),  // TODO reduce oversubscription
                        cuco::sentinel::empty_key{empty_node_index_sentinel},
                        cuco::sentinel::empty_value{empty_node_index_sentinel},
                        cuco::sentinel::erased_key{-2},
                        hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value()};
  // path compression is not used since extra writes make all map operations slow.
  auto const d_hasher = [node_level      = node_levels.begin(),
                         node_type       = node_type.begin(),
                         parent_node_ids = parent_node_ids.begin()] __device__(auto node_id) {
    auto hash =
      cudf::detail::hash_combine(cudf::detail::default_hash<TreeDepthT>{}(node_level[node_id]),
                                 cudf::detail::default_hash<size_type>{}(node_type[node_id]));
    node_id = parent_node_ids[node_id];
    while (node_id != parent_node_sentinel) {
      hash = cudf::detail::hash_combine(
        hash, cudf::detail::default_hash<TreeDepthT>{}(node_level[node_id]));
      hash = cudf::detail::hash_combine(
        hash, cudf::detail::default_hash<size_type>{}(node_type[node_id]));
      node_id = parent_node_ids[node_id];
    }
    return hash;
  };

  rmm::device_uvector<hash_value_type> node_hash(num_nodes, stream);
  thrust::tabulate(rmm::exec_policy(stream), node_hash.begin(), node_hash.end(), d_hasher);
  auto const d_hashed_cache = [node_hash = node_hash.begin()] __device__(auto node_id) {
    return node_hash[node_id];
  };

  auto const d_equal = [node_level      = node_levels.begin(),
                        node_type       = node_type.begin(),
                        parent_node_ids = parent_node_ids.begin(),
                        d_hashed_cache] __device__(auto node_id1, auto node_id2) {
    if (node_id1 == node_id2) return true;
    if (d_hashed_cache(node_id1) != d_hashed_cache(node_id2)) return false;
    auto const is_equal_level = [node_level, node_type](auto node_id1, auto node_id2) {
      if (node_id1 == node_id2) return true;
      return node_level[node_id1] == node_level[node_id2] and
             node_type[node_id1] == node_type[node_id2];
    };
    // if both nodes have same node types at all levels, it will check until it has common parent
    // or root.
    while (node_id1 != parent_node_sentinel and node_id2 != parent_node_sentinel and
           node_id1 != node_id2 and is_equal_level(node_id1, node_id2)) {
      node_id1 = parent_node_ids[node_id1];
      node_id2 = parent_node_ids[node_id2];
    }
    return node_id1 == node_id2;
  };

  // insert and convert node ids to unique set ids
  auto const num_inserted = thrust::count_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(num_nodes),
    [d_hashed_cache,
     d_equal,
     view       = key_map.get_device_mutable_view(),
     uq_node_id = col_id.begin()] __device__(auto node_id) mutable {
      auto it = view.insert_and_find(cuco::make_pair(node_id, node_id), d_hashed_cache, d_equal);
      uq_node_id[node_id] = (it.first)->first.load(cuda::std::memory_order_relaxed);
      return it.second;
    });

  auto const num_columns = num_inserted;  // key_map.get_size() is not updated.
  rmm::device_uvector<size_type> unique_keys(num_columns, stream);
  key_map.retrieve_all(unique_keys.begin(), thrust::make_discard_iterator(), stream.value());

  return {std::move(col_id), std::move(unique_keys)};
}

/**
 * @brief Generates column id and parent column id for each node
 *
 * 1. Generate col_id:
 *    a. Set operation on entire path of each node, translate each node id to set id.
 *       (two level hashing)
 *    b. gather unique set ids.
 *    c. sort and use binary search to generate column ids.
 *    d. Translate parent node ids to parent column ids.
 *
 * All inputs and outputs are in node_id order.
 * @param d_input JSON string in device memory
 * @param d_tree Tree representation of the JSON
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return column_id, parent_column_id
 */
std::pair<rmm::device_uvector<NodeIndexT>, rmm::device_uvector<NodeIndexT>> generate_column_id(
  device_span<SymbolT const> d_input,
  tree_meta_t const& d_tree,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto const num_nodes = d_tree.node_categories.size();

  // Two level hashing:
  //   one for field names -> node_type and,
  //   another for {node_level, node_category} + field hash for the entire path
  //    which is {node_level, node_type} recursively using parent_node_id
  auto [col_id, unique_keys] = [&]() {
    // Convert node_category + field_name to node_type.
    rmm::device_uvector<size_type> node_type =
      hash_node_type_with_field_name(d_input, d_tree, stream);

    // hash entire path from node to root.
    return hash_node_path(d_tree.node_levels, node_type, d_tree.parent_node_ids, stream, mr);
  }();

  thrust::sort(rmm::exec_policy(stream), unique_keys.begin(), unique_keys.end());
  thrust::lower_bound(rmm::exec_policy(stream),
                      unique_keys.begin(),
                      unique_keys.end(),
                      col_id.begin(),
                      col_id.end(),
                      col_id.begin());

  rmm::device_uvector<size_type> parent_col_id(num_nodes, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    d_tree.parent_node_ids.begin(),
                    d_tree.parent_node_ids.end(),
                    parent_col_id.begin(),
                    [col_id = col_id.begin()] __device__(auto node_id) {
                      return node_id >= 0 ? col_id[node_id] : parent_node_sentinel;
                    });

  return {std::move(col_id), std::move(parent_col_id)};
}

/**
 * @brief Computes row indices of each node in the hierarchy.
 * 2. Generate row_offset.
 *   a. Extract only list children
 *   b. stable_sort by parent_col_id.
 *   c. scan_by_key {parent_col_id} (done only on nodes who's parent is list)
 *   d. propagate to non-list leaves from parent list node by recursion
 *
 * pre-condition:
 *  d_tree.node_categories, d_tree.parent_node_ids, parent_col_id are in order of node_id.
 * post-condition: row_offsets is in order of node_id.
 *  parent_col_id is moved and reused inside this function.
 * @param parent_col_id parent node's column id
 * @param d_tree Tree representation of the JSON string
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return row_offsets
 */
rmm::device_uvector<size_type> compute_row_offsets(rmm::device_uvector<NodeIndexT>&& parent_col_id,
                                                   tree_meta_t const& d_tree,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto const num_nodes = d_tree.node_categories.size();

  rmm::device_uvector<size_type> scatter_indices(num_nodes, stream);
  thrust::sequence(rmm::exec_policy(stream), scatter_indices.begin(), scatter_indices.end());

  // Extract only list children. (nodes who's parent is a list/root)
  auto const list_parent_end =
    thrust::remove_if(rmm::exec_policy(stream),
                      thrust::make_zip_iterator(parent_col_id.begin(), scatter_indices.begin()),
                      thrust::make_zip_iterator(parent_col_id.end(), scatter_indices.end()),
                      d_tree.parent_node_ids.begin(),
                      [node_categories = d_tree.node_categories.begin()] __device__(auto pnid) {
                        return !(pnid == parent_node_sentinel || node_categories[pnid] == NC_LIST);
                      });
  auto const num_list_parent = thrust::distance(
    thrust::make_zip_iterator(parent_col_id.begin(), scatter_indices.begin()), list_parent_end);

  thrust::stable_sort_by_key(rmm::exec_policy(stream),
                             parent_col_id.begin(),
                             parent_col_id.begin() + num_list_parent,
                             scatter_indices.begin());

  rmm::device_uvector<size_type> row_offsets(num_nodes, stream, mr);
  // TODO is it possible to generate list child_offsets too here?
  // write only 1st child offset to parent node id child_offsets?
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                parent_col_id.begin(),
                                parent_col_id.begin() + num_list_parent,
                                thrust::make_constant_iterator<size_type>(1),
                                row_offsets.begin());

  // Using scatter instead of sort.
  auto& temp_storage = parent_col_id;  // reuse parent_col_id as temp storage
  thrust::scatter(rmm::exec_policy(stream),
                  row_offsets.begin(),
                  row_offsets.begin() + num_list_parent,
                  scatter_indices.begin(),
                  temp_storage.begin());
  row_offsets = std::move(temp_storage);

  // Propagate row offsets to non-list leaves from list's immediate children node by recursion
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
      auto const parent_node_id = parent_node_ids[node_id];
      return parent_node_id != parent_node_sentinel and
             !(node_categories[parent_node_id] == node_t::NC_LIST);
    });
  return row_offsets;
}

// This algorithm assigns a unique column id to each node in the tree.
// The row offset is the row index of the node in that column id.
// Algorithm:
// 1. Generate col_id:
//   a. Set operation on entire path of each node, translate each node id to set id.
//   b. gather unique set ids.
//   c. sort and use binary search to generate column ids.
//   d. Translate parent node ids to parent column ids.
// 2. Generate row_offset.
//   a. filter only list children
//   a. stable_sort by parent_col_id.
//   b. scan_by_key {parent_col_id} (done only on nodes whose parent is a list)
//   c. propagate to non-list leaves from parent list node by recursion
std::tuple<rmm::device_uvector<NodeIndexT>, rmm::device_uvector<size_type>>
records_orient_tree_traversal(device_span<SymbolT const> d_input,
                              tree_meta_t const& d_tree,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto [new_col_id, new_parent_col_id] = generate_column_id(d_input, d_tree, stream, mr);

  auto row_offsets = compute_row_offsets(std::move(new_parent_col_id), d_tree, stream, mr);
  return std::tuple{std::move(new_col_id), std::move(row_offsets)};
}

}  // namespace detail
}  // namespace cudf::io::json
