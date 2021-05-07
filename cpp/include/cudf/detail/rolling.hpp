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

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @copydoc std::unique_ptr<column> rolling_window(
 *            column_view const& input,
 *            column_view const& preceding_window,
 *            column_view const& following_window,
 *            size_type min_periods,
 *            rolling_aggregation const& agg,
 *            rmm::mr::device_memory_resource* mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> rolling_window(
  column_view const& input,
  column_view const& preceding_window,
  column_view const& following_window,
  size_type min_periods,
  rolling_aggregation const& agg,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace cudf
