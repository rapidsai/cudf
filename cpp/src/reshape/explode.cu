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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <memory>
#include <type_traits>

namespace cudf {
namespace detail {
namespace {
using GatherType   = rmm::device_uvector<size_type>;
using PositionType = rmm::device_uvector<size_type>;

std::unique_ptr<table> build_table(table_view const& input_table,
                                   size_type const explode_column_idx,
                                   column_view const& sliced_child,
                                   GatherType const& gather_map,
                                   GatherType const& explode_col_gather_map,
                                   PositionType position_array,
                                   bool include_position,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  auto select_iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [explode_column_idx](size_type i) { return i >= explode_column_idx ? i + 1 : i; });
  std::vector<size_type> selected_columns(select_iter, select_iter + input_table.num_columns() - 1);

  auto gathered_table = detail::gather(input_table.select(selected_columns),
                                       gather_map.begin(),
                                       gather_map.end(),
                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                       stream,
                                       mr);

  std::vector<std::unique_ptr<column>> columns = gathered_table.release()->release();

  columns.insert(columns.begin() + explode_column_idx,
                 explode_col_gather_map.size() > 0
                   ? std::move(detail::gather(table_view({sliced_child}),
                                              explode_col_gather_map.begin(),
                                              explode_col_gather_map.end(),
                                              cudf::out_of_bounds_policy::NULLIFY,
                                              stream,
                                              mr)
                                 ->release()[0])
                   : std::make_unique<column>(sliced_child, stream, mr));

  if (include_position) {
    size_type position_size = position_array.size();
    columns.insert(columns.begin() + explode_column_idx,
                   std::make_unique<column>(
                     data_type(type_to_id<size_type>()), position_size, position_array.release()));
  }

  return std::make_unique<table>(std::move(columns));
}
}  // namespace

std::unique_ptr<table> explode(table_view const& input_table,
                               size_type const explode_column_idx,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  lists_column_view explode_col{input_table.column(explode_column_idx)};
  auto sliced_child = explode_col.get_sliced_child(stream);
  GatherType gather_map(sliced_child.size(), stream);

  // Sliced columns may require rebasing of the offsets.
  auto offsets = explode_col.offsets_begin();
  // offsets + 1 here to skip the 0th offset, which removes a - 1 operation later.
  auto offsets_minus_one = thrust::make_transform_iterator(
    offsets + 1, [offsets] __device__(auto i) { return (i - offsets[0]) - 1; });
  auto counting_iter = thrust::make_counting_iterator(0);

  // This looks like an off-by-one bug, but what is going on here is that we need to reduce each
  // result from `lower_bound` by 1 to build the correct gather map. This can be accomplished by
  // skipping the first entry and using the result of `lower_bound` directly.
  thrust::lower_bound(rmm::exec_policy(stream),
                      offsets_minus_one,
                      offsets_minus_one + explode_col.size(),
                      counting_iter,
                      counting_iter + gather_map.size(),
                      gather_map.begin());

  GatherType empty{0, stream};
  PositionType empty_pos{0, stream};
  return build_table(input_table,
                     explode_column_idx,
                     sliced_child,
                     gather_map,
                     empty,
                     std::move(empty_pos),
                     false,
                     stream,
                     mr);
}

std::unique_ptr<table> explode_position(table_view const& input_table,
                                        size_type const explode_column_idx,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  lists_column_view explode_col{input_table.column(explode_column_idx)};
  auto sliced_child = explode_col.get_sliced_child(stream);
  GatherType gather_map(sliced_child.size(), stream);

  // Sliced columns may require rebasing of the offsets.
  auto offsets = explode_col.offsets_begin();
  // offsets + 1 here to skip the 0th offset, which removes a - 1 operation later.
  auto offsets_minus_one = thrust::make_transform_iterator(
    offsets + 1, [offsets] __device__(auto i) { return (i - offsets[0]) - 1; });
  auto counting_iter = thrust::make_counting_iterator(0);

  PositionType pos(sliced_child.size(), stream, mr);

  // This looks like an off-by-one bug, but what is going on here is that we need to reduce each
  // result from `lower_bound` by 1 to build the correct gather map. This can be accomplished by
  // skipping the first entry and using the result of `lower_bound` directly.
  thrust::transform(rmm::exec_policy(stream),
                    counting_iter,
                    counting_iter + gather_map.size(),
                    gather_map.begin(),
                    [position_array = pos.data(),
                     offsets_minus_one,
                     offsets,
                     offset_size = explode_col.size()] __device__(auto idx) -> size_type {
                      auto lb_idx =
                        thrust::lower_bound(
                          thrust::seq, offsets_minus_one, offsets_minus_one + offset_size, idx) -
                        offsets_minus_one;
                      position_array[idx] = idx - (offsets[lb_idx] - offsets[0]);
                      return lb_idx;
                    });

  GatherType empty{0, stream};
  return build_table(input_table,
                     explode_column_idx,
                     sliced_child,
                     gather_map,
                     empty,
                     std::move(pos),
                     true,
                     stream,
                     mr);
}

std::unique_ptr<table> explode_outer(table_view const& input_table,
                                     size_type const explode_column_idx,
                                     bool include_position,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  lists_column_view explode_col{input_table.column(explode_column_idx)};
  auto sliced_child  = explode_col.get_sliced_child(stream);
  auto counting_iter = thrust::make_counting_iterator(0);
  auto offsets       = explode_col.offsets_begin();

  // number of nulls or empty lists found so far in the explode column
  GatherType null_offset(explode_col.size(), stream);

  auto null_or_empty = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [offsets, offsets_size = explode_col.size() - 1] __device__(int idx) {
      return (idx > offsets_size || (offsets[idx + 1] != offsets[idx])) ? 0 : 1;
    });
  thrust::inclusive_scan(
    rmm::exec_policy(stream), null_iter, null_iter + sliced_child.size(), null_offset.begin());

  auto null_or_empty_count = null_offset.size() > 0 ? null_offset.back_element(stream) : 0;
  if (null_or_empty_count == 0) {
    // performance penalty, run simple explode
    return include_position ? explode_position(input_table, explode_column_idx, stream, mr)
                            : explode(input_table, explode_column_idx, stream, mr);
  }

  auto gather_map_size = sliced_child.size() + null_or_empty_count;

  GatherType gather_map(gather_map_size, stream);
  GatherType explode_col_gather_map(gather_map_size, stream);
  PositionType pos(include_position ? gather_map_size : 0, stream, mr);

  // offsets + 1 here to skip the 0th offset, which removes a - 1 operation later.
  auto offsets_minus_one = thrust::make_transform_iterator(
    thrust::next(offsets), [offsets] __device__(auto i) { return (i - offsets[0]) - 1; });
  // Fill in gather map with all the child column's entries
  thrust::for_each(rmm::exec_policy(stream),
                   counting_iter,
                   counting_iter + sliced_child.size(),
                   [offsets_minus_one,
                    gather_map             = gather_map.begin(),
                    explode_col_gather_map = explode_col_gather_map.begin(),
                    position_array         = pos.begin(),
                    include_position,
                    offsets,
                    null_offset = null_offset.begin(),
                    null_iter,
                    offset_size = explode_col.offsets().size() - 1] __device__(auto idx) {
                     auto lb_idx =
                       thrust::lower_bound(
                         thrust::seq, offsets_minus_one, offsets_minus_one + (offset_size), idx) -
                       offsets_minus_one;
                     auto index_to_write                    = null_offset[lb_idx] + idx;
                     gather_map[index_to_write]             = lb_idx;
                     explode_col_gather_map[index_to_write] = idx;
                     if (include_position) {
                       position_array[index_to_write] = idx - (offsets[lb_idx] - offsets[0]);
                     }
                     if (null_iter[idx] == 1) {
                       auto invalid_index =
                         null_offset[idx] == 0 ? offsets[idx] : offsets[idx] + null_offset[idx] - 1;
                       gather_map[invalid_index] = idx;

                       // negative one to indicate a null value
                       explode_col_gather_map[invalid_index] = -1;

                       if (include_position) { position_array[invalid_index] = 0; }
                     }
                   });

  return build_table(input_table,
                     explode_column_idx,
                     sliced_child,
                     gather_map,
                     explode_col_gather_map,
                     std::move(pos),
                     include_position,
                     stream,
                     mr);
}

}  // namespace detail

/**
 * @copydoc cudf::explode(input_table,explode_column_idx,rmm::mr::device_memory_resource)
 */
std::unique_ptr<table> explode(table_view const& input_table,
                               size_type explode_column_idx,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input_table.column(explode_column_idx).type().id() == type_id::LIST,
               "Unsupported non-list column");
  return detail::explode(input_table, explode_column_idx, rmm::cuda_stream_default, mr);
}

/**
 * @copydoc cudf::explode_position(input_table,explode_column_idx,rmm::mr::device_memory_resource)
 */
std::unique_ptr<table> explode_position(table_view const& input_table,
                                        size_type explode_column_idx,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input_table.column(explode_column_idx).type().id() == type_id::LIST,
               "Unsupported non-list column");
  return detail::explode_position(input_table, explode_column_idx, rmm::cuda_stream_default, mr);
}

/**
 * @copydoc cudf::explode_outer(input_table,explode_column_idx,rmm::mr::device_memory_resource)
 */
std::unique_ptr<table> explode_outer(table_view const& input_table,
                                     size_type explode_column_idx,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input_table.column(explode_column_idx).type().id() == type_id::LIST,
               "Unsupported non-list column");
  return detail::explode_outer(
    input_table, explode_column_idx, false, rmm::cuda_stream_default, mr);
}

/**
 * @copydoc
 * cudf::explode_outer_position(input_table,explode_column_idx,rmm::mr::device_memory_resource)
 */
std::unique_ptr<table> explode_outer_position(table_view const& input_table,
                                              size_type explode_column_idx,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input_table.column(explode_column_idx).type().id() == type_id::LIST,
               "Unsupported non-list column");
  return detail::explode_outer(input_table, explode_column_idx, true, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
