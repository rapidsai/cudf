/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace detail {

namespace {
/**
 * @brief The enum specifying which sorting method to use (stable or unstable).
 */
enum class sort_method { STABLE, UNSTABLE };

// returns segment indices for each element for all segments.
// first segment begin index = 0, last segment end index = num_rows.
rmm::device_uvector<size_type> get_segment_indices(size_type num_rows,
                                                   column_view const& offsets,
                                                   rmm::cuda_stream_view stream)
{
  rmm::device_uvector<size_type> segment_ids(num_rows, stream);

  auto offset_begin = offsets.begin<size_type>();  // assumes already offset column contains offset.
  auto offsets_minus_one = thrust::make_transform_iterator(
    offset_begin, [offset_begin] __device__(auto i) { return i - 1; });
  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  thrust::lower_bound(rmm::exec_policy(stream),
                      offsets_minus_one,
                      offsets_minus_one + offsets.size(),
                      counting_iter,
                      counting_iter + segment_ids.size(),
                      segment_ids.begin());
  return segment_ids;
}

std::unique_ptr<column> segmented_sorted_order_common(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  sort_method sorting,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(segment_offsets.type() == data_type(type_to_id<size_type>()),
               "segment offsets should be size_type");
  // Get segment id of each element in all segments.
  auto segment_ids = get_segment_indices(keys.num_rows(), segment_offsets, stream);

  // insert segment id before all columns.
  std::vector<column_view> keys_with_segid;
  keys_with_segid.reserve(keys.num_columns() + 1);
  keys_with_segid.push_back(
    column_view(data_type(type_to_id<size_type>()), segment_ids.size(), segment_ids.data()));
  keys_with_segid.insert(keys_with_segid.end(), keys.begin(), keys.end());
  auto segid_keys = table_view(keys_with_segid);

  auto prepend_default = [](auto const& vector, auto default_value) {
    if (vector.empty()) return vector;
    std::remove_cv_t<std::remove_reference_t<decltype(vector)>> pre_vector;
    pre_vector.reserve(pre_vector.size() + 1);
    pre_vector.push_back(default_value);
    pre_vector.insert(pre_vector.end(), vector.begin(), vector.end());
    return pre_vector;
  };
  auto child_column_order    = prepend_default(column_order, order::ASCENDING);
  auto child_null_precedence = prepend_default(null_precedence, null_order::AFTER);

  // return sorted order of child columns
  return sorting == sort_method::STABLE
           ? detail::stable_sorted_order(
               segid_keys, child_column_order, child_null_precedence, stream, mr)
           : detail::sorted_order(
               segid_keys, child_column_order, child_null_precedence, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key_common(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    sort_method sorting,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(values.num_rows() == keys.num_rows(),
               "Mismatch in number of rows for values and keys");
  auto sorted_order = sorting == sort_method::STABLE
                        ? stable_segmented_sorted_order(keys,
                                                        segment_offsets,
                                                        column_order,
                                                        null_precedence,
                                                        stream,
                                                        rmm::mr::get_current_device_resource())
                        : segmented_sorted_order(keys,
                                                 segment_offsets,
                                                 column_order,
                                                 null_precedence,
                                                 stream,
                                                 rmm::mr::get_current_device_resource());

  // Gather segmented sort of child value columns`
  return detail::gather(values,
                        sorted_order->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

}  // namespace

std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  return segmented_sorted_order_common(
    keys, segment_offsets, column_order, null_precedence, sort_method::UNSTABLE, stream, mr);
}

std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return segmented_sorted_order_common(
    keys, segment_offsets, column_order, null_precedence, sort_method::STABLE, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  return segmented_sort_by_key_common(values,
                                      keys,
                                      segment_offsets,
                                      column_order,
                                      null_precedence,
                                      sort_method::UNSTABLE,
                                      stream,
                                      mr);
}

std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  return segmented_sort_by_key_common(
    values, keys, segment_offsets, column_order, null_precedence, sort_method::STABLE, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sorted_order(
    keys, segment_offsets, column_order, null_precedence, cudf::default_stream_value, mr);
}

std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_segmented_sorted_order(
    keys, segment_offsets, column_order, null_precedence, cudf::default_stream_value, mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sort_by_key(
    values, keys, segment_offsets, column_order, null_precedence, cudf::default_stream_value, mr);
}

std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_segmented_sort_by_key(
    values, keys, segment_offsets, column_order, null_precedence, cudf::default_stream_value, mr);
}

}  // namespace cudf
