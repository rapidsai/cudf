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
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/table/experimental/row_operators.cuh>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf::lists {
namespace detail {

namespace {

/**
 * @brief Create a hash map with keys are indices of all elements in the input column.
 */
std::unique_ptr<cudf::detail::hash_map_type> create_map(column_view const& input,
                                                        rmm::cuda_stream_view stream)
{
  auto map = std::make_unique<cudf::detail::hash_map_type>(
    compute_hash_table_size(input.size()),
    cudf::detail::COMPACTION_EMPTY_KEY_SENTINEL,
    cudf::detail::COMPACTION_EMPTY_VALUE_SENTINEL,
    cudf::detail::hash_table_allocator_type{default_allocator<char>{}, stream},
    stream.value());

  auto const input_tv = table_view{{input}};
  auto const hasher   = cudf::experimental::row::hash::row_hasher(input_tv, stream);
  auto const d_hasher = cudf::detail::experimental::compaction_hash(
    hasher.device_hasher(nullate::DYNAMIC{input.has_nulls()}));
  auto const kv_it = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(size_type const i) { return cuco::make_pair(i, i); });

  map->insert(kv_it, kv_it + input.size(), d_hasher, thrust::equal_to<size_type>{}, stream.value());
  return map;
}

/**
 * @brief Check the existence of rows in the rhs column in the hash map, which was created by rows
 *        of the lhs column.
 */
// todo: keys must be table with the first col is labels
auto check_contains(std::unique_ptr<cudf::detail::hash_map_type> const& map,
                    column_view const& lhs,
                    column_view const& rhs,
                    rmm::cuda_stream_view stream)
{
  auto contained = rmm::device_uvector<bool>(rhs.size(), stream);

  auto const lhs_tv   = table_view{{lhs}};
  auto const rhs_tv   = table_view{{rhs}};
  auto const hasher   = cudf::experimental::row::hash::row_hasher(rhs_tv, stream);
  auto const d_hasher = cudf::experimental::row::hash::index_normalized_hasher_adapter{
    cudf::detail::experimental::compaction_hash(
      hasher.device_hasher(nullate::DYNAMIC{rhs.has_nulls()}))};
  auto const comparator =
    cudf::experimental::row::equality::two_table_comparator(lhs_tv, rhs_tv, stream);
  auto const d_eqcomp = cudf::experimental::row::equality::index_normalized_comparator_adapter{
    comparator.equal_to(nullate::DYNAMIC{lhs.has_nulls() || rhs.has_nulls()})};
  auto const rhs_it = thrust::make_reverse_iterator(thrust::make_counting_iterator(size_type{0}));

  map->contains(rhs_it, rhs_it + rhs.size(), contained.begin(), d_hasher, d_eqcomp, stream.value());
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
 * @brief Extract rows from the input data column and labels array based on the boolean values in
 *        the input `condition` column.
 */
auto extract_if(column_view const& data,
                rmm::device_uvector<size_type> const& labels,
                rmm::device_uvector<bool> const& condition,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr)

{
  CUDF_EXPECTS(static_cast<std::size_t>(data.size()) == labels.size(), "TBA");

  auto gather_map = rmm::device_uvector<size_type>(data.size(), stream);
  auto out_labels = rmm::device_uvector<size_type>(labels.size(), stream);

  auto const input_it =
    thrust::make_zip_iterator(thrust::make_counting_iterator<size_type>(0), labels.begin());
  auto const output_it = thrust::make_zip_iterator(gather_map.begin(), out_labels.begin());
  auto const copy_end  = thrust::copy_if(rmm::exec_policy(stream),
                                        input_it,
                                        input_it + data.size(),
                                        condition.begin(),
                                        output_it,
                                        thrust::identity{});

  auto const gather_map_size = thrust::distance(output_it, copy_end);
  auto out_data              = cudf::detail::gather(table_view{{data}},
                                       gather_map.begin(),
                                       gather_map.begin() + gather_map_size,
                                       out_of_bounds_policy::DONT_CHECK,
                                       stream,
                                       mr);

  return std::pair(std::move(out_data->release().front()), std::move(out_labels));
}

}  // namespace

std::unique_ptr<column> set_overlap(lists_column_view const& lhs,
                                    lists_column_view const& rhs,
                                    null_equality nulls_equal,
                                    nan_equality nans_equal,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  // - Insert lhs child elements into map.
  // - Check contains for rhs child elements.
  // - Generate labels for rhs child elements.
  // - `reduce_by_key` with `logical_or` functor and keys are labels, values are contains.

  auto const lhs_child = lhs.get_sliced_child(stream);
  auto const rhs_child = rhs.get_sliced_child(stream);
  auto const map       = create_map(lhs_child, stream);
  auto const contained = check_contains(map, lhs_child, rhs_child, stream);
  auto const labels    = generate_labels(rhs_child, stream);

  // This stores the unique label values, used as scatter map.
  auto list_indices = rmm::device_uvector<size_type>(lhs.size(), stream);

  // Stores the overlap check for non-empty lists.
  auto overlap_result = rmm::device_uvector<bool>(lhs.size(), stream);

  auto const end                    = thrust::reduce_by_key(rmm::exec_policy(stream),
                                         labels.begin(),
                                         labels.end(),
                                         contained.begin(),
                                         list_indices.begin(),
                                         overlap_result.begin(),
                                         thrust::equal_to{},
                                         thrust::logical_or{});
  auto const num_non_empty_segments = thrust::distance(overlap_result.begin(), end.second);

  // todo fix null mask null count
  auto result = make_numeric_column(data_type{type_to_id<bool>()},
                                    lhs.size(),
                                    copy_bitmask(lhs.parent()),
                                    lhs.null_count(),
                                    stream,
                                    mr);

  thrust::scatter(rmm::exec_policy(stream),
                  overlap_result.begin(),
                  overlap_result.begin() + num_non_empty_segments,
                  list_indices.begin(),
                  result->mutable_view().template begin<bool>());

  return result;
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      null_equality nulls_equal,
                                      nan_equality nans_equal,
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

  auto const lhs_child = lhs.get_sliced_child(stream);
  auto const rhs_child = rhs.get_sliced_child(stream);
  auto const map       = create_map(lhs_child, stream);
  auto const contained = check_contains(map, lhs_child, rhs_child, stream);
  auto const labels    = generate_labels(rhs_child, stream);

  auto [out_child, intersect_labels] = extract_if(rhs_child, labels, contained, stream, mr);
  auto out_offsets = reconstruct_offsets(intersect_labels, lhs.size(), stream, mr);

  // todo : fix null
  return make_lists_column(
    lhs.size(), std::move(out_offsets), std::move(out_child), 0, {}, stream, mr);
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  null_equality nulls_equal,
                                  nan_equality nans_equal,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  // - concatenate_row(lhs, set_except(rhs, lhs))
  // - Alternative: concatenate_row(lhs, rhs) then `drop_list_duplicates`, however,
  //   `drop_list_duplicates` currently doesn't support nested types.
  // todo: add stream in detail version
  // fix concatenate_rows params.
  auto const diff = set_difference(rhs, lhs, nulls_equal, nans_equal, mr);
  return lists::concatenate_rows(table_view{{lhs.parent(), diff->view()}});
}

std::unique_ptr<column> set_difference(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
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

  auto const lhs_child     = lhs.get_sliced_child(stream);
  auto const rhs_child     = rhs.get_sliced_child(stream);
  auto const map           = create_map(rhs_child, stream);
  auto const inv_contained = [&] {
    auto contained = check_contains(map, rhs_child, lhs_child, stream);
    thrust::transform(rmm::exec_policy(stream),
                      contained.begin(),
                      contained.end(),
                      contained.begin(),
                      thrust::logical_not{});
    return contained;
  }();

  auto const labels = generate_labels(lhs_child, stream);

  auto [out_child, except_labels] = extract_if(lhs_child, labels, inv_contained, stream, mr);
  auto out_offsets                = reconstruct_offsets(except_labels, lhs.size(), stream, mr);

  // todo : fix null
  return make_lists_column(
    lhs.size(), std::move(out_offsets), std::move(out_child), 0, {}, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> set_overlap(lists_column_view const& lhs,
                                    lists_column_view const& rhs,
                                    null_equality nulls_equal,
                                    nan_equality nans_equal,
                                    rmm::mr::device_memory_resource* mr)
{
  return detail::set_overlap(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      null_equality nulls_equal,
                                      nan_equality nans_equal,
                                      rmm::mr::device_memory_resource* mr)
{
  return detail::set_intersect(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  null_equality nulls_equal,
                                  nan_equality nans_equal,
                                  rmm::mr::device_memory_resource* mr)
{
  return detail::set_union(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_difference(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::mr::device_memory_resource* mr)
{
  return detail::set_difference(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
