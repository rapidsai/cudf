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

#include <algorithm>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <cub/device/device_segmented_radix_sort.cuh>
#include <iterator>
#include <memory>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace detail {

// returns row index for each element of the row in list column.
rmm::device_uvector<size_type> get_list_segment_indices(lists_column_view const& lc,
                                                        rmm::cuda_stream_view stream)
{
  auto sliced_child = lc.get_sliced_child(stream);
  rmm::device_uvector<size_type> segment_ids(sliced_child.size(), stream);

  auto offsets           = lc.offsets().begin<size_type>() + lc.offset();
  auto offsets_minus_one = thrust::make_transform_iterator(
    offsets, [offsets] __device__(auto i) { return i - offsets[0] - 1; });
  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  thrust::lower_bound(rmm::exec_policy(stream),
                      offsets_minus_one + 1,
                      offsets_minus_one + lc.size() + 1,
                      counting_iter,
                      counting_iter + segment_ids.size(),
                      segment_ids.begin());
  return std::move(segment_ids);
}

std::unique_ptr<column> segmented_sorted_order(table_view const& values,
                                               table_view const& keys,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(std::all_of(keys.begin(),
                           keys.end(),
                           [](column_view const& col) { return col.type().id() == type_id::LIST; }),
               "segmented_sort only supports lists columns");
  CUDF_EXPECTS(std::all_of(values.begin(),
                           values.end(),
                           [](column_view const& col) { return col.type().id() == type_id::LIST; }),
               "segmented_sort only supports lists columns");
  // TODO check if all list sizes are equal. OR all offsets are equal (may be wrong).

  auto segment_ids = get_list_segment_indices(lists_column_view{keys.column(0)}, stream);
  // insert segment id before all child columns.
  std::vector<column_view> child_key_columns(keys.num_columns() + 1);
  child_key_columns[0] =
    column_view(data_type(type_to_id<size_type>()), segment_ids.size(), segment_ids.data());
  std::transform(keys.begin(), keys.end(), child_key_columns.begin() + 1, [stream](auto col) {
    return lists_column_view(col).get_sliced_child(stream);
  });
  auto child_keys = table_view(child_key_columns);

  std::vector<order> child_column_order(column_order);
  if (not column_order.empty())
    child_column_order.insert(child_column_order.begin(), order::ASCENDING);
  std::vector<null_order> child_null_precedence(null_precedence);
  if (not null_precedence.empty())
    child_null_precedence.insert(child_null_precedence.begin(), null_order::AFTER);

  // create table_view of child columns
  return detail::sorted_order(child_keys, child_column_order, child_null_precedence, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto sorted_order = segmented_sorted_order(values, keys, column_order, null_precedence, stream);
  std::vector<column_view> child_columns(values.num_columns());
  std::transform(values.begin(), values.end(), child_columns.begin(), [stream](auto col) {
    return lists_column_view(col).get_sliced_child(stream);
  });
  // return std::unique_ptr<table>(new table{table_view{std::vector<column_view>{*sorted_order}}});
  // TODO build the list columns from returned table! and packit into table!
  auto child_result = detail::gather(table_view{child_columns},
                                     sorted_order->view(),
                                     out_of_bounds_policy::DONT_CHECK,
                                     detail::negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr)
                        ->release();

  std::vector<std::unique_ptr<column>> list_columns;
  std::transform(  // thrust::host,
    values.begin(),
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
                               std::move(null_mask));
    });
  return std::make_unique<table>(std::move(list_columns));
  // TODO write tests and verify */
}
}  // namespace detail

std::unique_ptr<table> segmented_sort(table_view input,
                                      std::vector<order> const& column_order,
                                      std::vector<null_order> const& null_precedence,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sort_by_key(
    input, input, column_order, null_precedence, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
