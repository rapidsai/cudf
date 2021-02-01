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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

#include <algorithm>
#include <iterator>
#include <memory>

namespace cudf {
namespace detail {

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
  return std::move(segment_ids);
}

void validate_list_columns(table_view const& keys, rmm::cuda_stream_view stream)
{
  // check if all are list columns
  CUDF_EXPECTS(std::all_of(keys.begin(),
                           keys.end(),
                           [](column_view const& col) { return col.type().id() == type_id::LIST; }),
               "segmented_sort_by_key only supports lists columns");
  // check if all list sizes are equal.
  auto table_device  = table_device_view::create(keys, stream);
  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  CUDF_EXPECTS(
    thrust::all_of(rmm::exec_policy(stream),
                   counting_iter,
                   counting_iter + keys.num_rows(),
                   [d_keys = *table_device] __device__(size_type idx) {
                     auto size = list_size_functor{d_keys.column(0)}(idx);
                     return thrust::all_of(
                       thrust::seq, d_keys.begin(), d_keys.end(), [&](auto const& d_column) {
                         return list_size_functor{d_column}(idx) == size;
                       });
                   }),
    "size of each list in a row of table should be same");
}

std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
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

  std::vector<order> child_column_order(column_order);
  if (not column_order.empty())
    child_column_order.insert(child_column_order.begin(), order::ASCENDING);
  std::vector<null_order> child_null_precedence(null_precedence);
  if (not null_precedence.empty())
    child_null_precedence.insert(child_null_precedence.begin(), null_order::AFTER);

  // return sorted order of child columns
  return detail::sorted_order(segid_keys, child_column_order, child_null_precedence, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(values.num_rows() == keys.num_rows(),
               "Mismatch in number of rows for values and keys");
  auto sorted_order = segmented_sorted_order(keys,
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

std::unique_ptr<table> sort_lists(table_view const& values,
                                  table_view const& keys,
                                  std::vector<order> const& column_order,
                                  std::vector<null_order> const& null_precedence,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(keys.num_columns() > 0, "keys table should have atleast one list column");
  std::vector<column_view> key_value_columns;
  key_value_columns.reserve(keys.num_columns() + values.num_columns());
  key_value_columns.insert(key_value_columns.end(), keys.begin(), keys.end());
  key_value_columns.insert(key_value_columns.end(), values.begin(), values.end());
  validate_list_columns(table_view{key_value_columns}, stream);

  // child columns of keys
  auto child_key_columns = thrust::make_transform_iterator(
    keys.begin(), [stream](auto col) { return lists_column_view(col).get_sliced_child(stream); });
  auto child_keys =
    table_view{std::vector<column_view>(child_key_columns, child_key_columns + keys.num_columns())};

  // segment offsets from first list column
  auto lc              = lists_column_view{keys.column(0)};
  auto offset          = lc.offsets();
  auto segment_offsets = cudf::detail::slice(offset, {lc.offset(), offset.size()}, stream)[0];
  // child columns of values
  auto child_value_columns = thrust::make_transform_iterator(
    values.begin(), [stream](auto col) { return lists_column_view(col).get_sliced_child(stream); });
  auto child_values = table_view{
    std::vector<column_view>(child_value_columns, child_value_columns + values.num_columns())};

  // Get segment sorted child columns of list columns
  auto child_result =
    segmented_sort_by_key(
      child_values, child_keys, segment_offsets, column_order, null_precedence, stream, mr)
      ->release();

  // Construct list columns from gathered child columns & return
  std::vector<std::unique_ptr<column>> list_columns;
  std::transform(values.begin(),
                 values.end(),
                 std::make_move_iterator(child_result.begin()),
                 std::back_inserter(list_columns),
                 [&stream, &mr](auto& input_list, auto&& sorted_child) {
                   auto output_offset =
                     std::make_unique<column>(lists_column_view(input_list).offsets(), stream, mr);
                   auto null_mask = cudf::detail::copy_bitmask(input_list, stream, mr);
                   // Assemble list column & return
                   return make_lists_column(input_list.size(),
                                            std::move(output_offset),
                                            std::move(sorted_child),
                                            input_list.null_count(),
                                            std::move(null_mask),
                                            stream,
                                            mr);
                 });
  return std::make_unique<table>(std::move(list_columns));
}
}  // namespace detail

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sort_by_key(
    values, keys, segment_offsets, column_order, null_precedence, rmm::cuda_stream_default, mr);
}
std::unique_ptr<table> sort_lists(table_view const& values,
                                  table_view const& keys,
                                  std::vector<order> const& column_order,
                                  std::vector<null_order> const& null_precedence,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_lists(
    values, keys, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
