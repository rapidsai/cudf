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

#include "range_utils.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <optional>

namespace CUDF_EXPORT cudf {
namespace detail {

[[nodiscard]] std::unique_ptr<column> make_following_range_window_bound(
  column_view const& orderby,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  order order,
  null_order null_order,
  range_window_type window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return make_range_window_bound<rolling::direction::FOLLOWING>(
    orderby, grouping, order, null_order, window, stream, mr);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
