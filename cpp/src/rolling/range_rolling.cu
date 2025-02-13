/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/rolling.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <optional>

namespace CUDF_EXPORT cudf {
namespace detail {

rmm::device_uvector<cudf::size_type> nulls_per_group(column_view const& orderby,
                                                     rmm::device_uvector<size_type> const& offsets,
                                                     rmm::cuda_stream_view stream)
{
  auto d_orderby        = column_device_view::create(orderby, stream);
  auto const num_groups = offsets.size() - 1;
  std::size_t bytes{0};
  auto is_null_it = cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0},
    cuda::proclaim_return_type<size_type>(
      [orderby = *d_orderby] __device__(size_type i) -> size_type {
        return static_cast<size_type>(orderby.is_null_nocheck(i));
      }));
  rmm::device_uvector<cudf::size_type> null_counts{num_groups, stream};
  cub::DeviceSegmentedReduce::Sum(nullptr,
                                  bytes,
                                  is_null_it,
                                  null_counts.begin(),
                                  num_groups,
                                  offsets.begin(),
                                  offsets.begin() + 1,
                                  stream.value());
  auto tmp = rmm::device_buffer(bytes, stream);
  cub::DeviceSegmentedReduce::Sum(tmp.data(),
                                  bytes,
                                  is_null_it,
                                  null_counts.begin(),
                                  num_groups,
                                  offsets.begin(),
                                  offsets.begin() + 1,
                                  stream.value());
  return null_counts;
}

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (group_keys.num_columns() > 0) {
    using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;
    sort_helper helper{group_keys, null_policy::INCLUDE, sorted::YES, {}};
    auto const& labels   = helper.group_labels(stream);
    auto const& offsets  = helper.group_offsets(stream);
    auto per_group_nulls = orderby.has_nulls() ? nulls_per_group(orderby, offsets, stream)
                                               : rmm::device_uvector<size_type>{0, stream};
    auto grouping = detail::rolling::preprocessed_group_info{labels, offsets, per_group_nulls};
    return {make_range_window<rolling::direction::PRECEDING>(
              orderby, grouping, order, null_order, preceding, stream, mr),
            make_range_window<rolling::direction::FOLLOWING>(
              orderby, grouping, order, null_order, preceding, stream, mr)};
  } else {
    return {make_range_window<rolling::direction::PRECEDING>(
              orderby, std::nullopt, order, null_order, preceding, stream, mr),
            make_range_window<rolling::direction::FOLLOWING>(
              orderby, std::nullopt, order, null_order, preceding, stream, mr)};
  }
}
}  // namespace detail

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    group_keys.num_columns() == 0 || group_keys.num_rows() == orderby.size(),
    "If a grouping table is provided, it must have same number of rows as the orderby column.");
  return detail::make_range_windows(
    group_keys, orderby, order, null_order, preceding, following, stream, mr);
}

}  // namespace CUDF_EXPORT cudf
