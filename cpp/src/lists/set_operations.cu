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

#include <stream_compaction/stream_compaction_common.cuh>
#include <stream_compaction/stream_compaction_common.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/lists/set_operations.hpp>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf::lists {
namespace detail {

namespace {

/**
 * @brief Create a hash map with keys are indices of all elements in the input column.
 */
auto create_map(column_view const& input, rmm::cuda_stream_view stream)
{
  auto map =
    detail::hash_map_type{compute_hash_table_size(input.size()),
                          detail::COMPACTION_EMPTY_KEY_SENTINEL,
                          detail::COMPACTION_EMPTY_VALUE_SENTINEL,
                          detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                          stream.value()};

  auto const input_tv = table_view{{input}};
  auto const hasher   = cudf::experimental::row::hash::row_hasher(input_tv, stream);
  auto const d_hasher = detail::experimental::compaction_hash(
    hasher.device_hasher(nullate::DYNAMIC{input.has_nulls()}));
  auto const kv_it = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(size_type const i) { return cuco::make_pair(i, i); });

  map.insert(kv_it, kv_it + input.size(), d_hasher, thrust::equal_to<size_type>{}, stream.value());
  return map;
}

/**
 * @brief Check the existence of rows in the keys column in the hash map.
 */
auto check_contains(detail::hash_map_type const& map,
                    bool map_has_nulls,
                    column_view const& keys,
                    rmm::cuda_stream_view stream)
{
  auto contained = rmm::device_uvector<bool>(keys.size(), stream);

  auto const keys_tv  = table_view{{keys}};
  auto const hasher   = cudf::experimental::row::hash::row_hasher(keys_tv, stream);
  auto const d_hasher = cudf::experimental::row::hash::negative_index_hasher_adapter{
    detail::experimental::compaction_hash(
      hasher.device_hasher(nullate::DYNAMIC{keys.has_nulls()}))};
  auto const comparator =
    cudf::experimental::row::equality::two_table_comparator(haystack_tv, keys_tv, stream);
  auto const d_eqcomp = cudf::experimental::row::equality::negative_index_comparator_adapter{
    comparator.device_comparator(nullate::DYNAMIC{map_has_nulls || keys.has_nulls()})};
  auto const keys_it = thrust::make_reverse_iterator(thrust::make_counting_iterator(size_type{0}));

  map.contains(
    keys_it, keys_it + keys.size(), contained.begin(), d_hasher, d_eqcomp, stream.value());
  return contained;
}

/**
 * @brief Generate labels for elements in the child column of the input lists column.
 * @param input
 */
auto generate_labels(lists_column_view const& input, rmm::cuda_stream_view stream)
{
  auto labels = rmm::device_uvector<size_type>(input.size(), stream);
  cudf::detail::label_segments(
    input.offsets_begin(), input.offsets_end(), labels.begin(), labels.end(), stream);
  return labels;
}

/**
 * @brief Reconstruct an offsets column from the input labels array.
 */
auto reconstruct_offsets(rmm::device_uvector<size_type> const& labels,
                         size_type n_rows,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)

{
  auto out_offsets = make_numeric_column(
    data_type{type_to_id<offset_type>()}, n_rows + 1, mask_state::UNALLOCATED, stream, mr);
  auto const offsets = out_offsets->mutable_view();
  cudf::detail::labels_to_offsets(labels.begin(),
                                  labels.end(),
                                  offsets.template begin<size_type>(),
                                  offsets.template end<size_type>(),
                                  stream);
  return out_offsets;
}

/**
 * @brief Extract rows from the input table based on the boolean values in the input `condition`
 * column.
 */
auto extract_if()

{
  //
}

}  // namespace

std::unique_ptr<column> overlap(lists_column_view const& lhs,
                                lists_column_view const& rhs,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  // - Insert lhs child elements into map.
  // - Check contains for rhs child elements.
  // - Generate labels for rhs child elements.
  // - `reduce_by_key` with `logical_or` functor and keys are labels, values are contains.
  return nullptr;
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  // - Insert lhs child elements into map.
  // - Check contains for rhs child element.
  // - Generate labels for rhs child elements.
  // - copy_if {indices, labels} for rhs child elements using contains conditions to {gather_map,
  //   intersect_labels}.
  // - output_child = pull rhs child elements from gather_map.
  // - output_offsets = reconstruct offsets from intersect_labels.
  // - return lists_column(output_child, output_offsets)
  return nullptr;
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  // - concatenate_row lhs and set_except(rhs, lhs)
  return nullptr;
}

std::unique_ptr<column> set_except(lists_column_view const& lhs,
                                   lists_column_view const& rhs,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  // - Insert rhs child elements.
  // - Check contains for lhs child element.
  // - Invert contains for lhs child element.
  // - Generate labels for lhs child elements.
  // - copy_if {indices, labels} using the inverted contains conditions to {gather_map,
  //   except_labels} for lhs child elements.
  // - Pull lhs child elements from gather_map.
  // - Reconstruct output offsets from except_labels for lhs.
  return nullptr;
}

}  // namespace detail

std::unique_ptr<column> overlap(lists_column_view const& lhs,
                                lists_column_view const& rhs,
                                rmm::mr::device_memory_resource* mr)
{
  return detail::overlap(lhs, rhs, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      rmm::mr::device_memory_resource* mr)
{
  return detail::set_intersect(lhs, rhs, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  rmm::mr::device_memory_resource* mr)
{
  return detail::set_union(lhs, rhs, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_except(lists_column_view const& lhs,
                                   lists_column_view const& rhs,
                                   rmm::mr::device_memory_resource* mr)
{
  return detail::set_except(lhs, rhs, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
