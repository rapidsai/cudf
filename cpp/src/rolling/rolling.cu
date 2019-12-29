/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/aggregation.hpp>
#include <rolling/rolling_detail.hpp>
#include <cudf/rolling.hpp>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/rolling/rolling.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/copying.hpp>

#include <rmm/device_scalar.hpp>

#include <thrust/binary_search.h>
#include <thrust/functional.h>

#include <memory>
#include <algorithm>

namespace cudf {
namespace experimental {

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS((preceding_window >= 0) && (following_window >= 0) && (min_periods >= 0),
               "Window sizes and min periods must be non-negative");

  auto preceding_window_begin = thrust::make_constant_iterator(preceding_window);
  auto following_window_begin = thrust::make_constant_iterator(following_window);

  return cudf::experimental::detail::rolling_window(input, preceding_window_begin,
                                                    following_window_begin, min_periods, aggr, mr, 0);
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr)
{
  if (preceding_window.size() == 0 || following_window.size() == 0) return empty_like(input);

  CUDF_EXPECTS(preceding_window.type().id() == INT32 && following_window.type().id() == INT32,
               "preceding_window/following_window must have INT32 type");

  CUDF_EXPECTS(preceding_window.size() == input.size() && following_window.size() == input.size(),
               "preceding_window/following_window size must match input size");

  return cudf::experimental::detail::rolling_window(input, preceding_window.begin<size_type>(),
                                                    following_window.begin<size_type>(),
                                                    min_periods, aggr, mr, 0);
}

} // namespace experimental 
} // namespace cudf