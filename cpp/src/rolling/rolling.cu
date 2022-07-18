/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "detail/rolling.cuh"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/iterator/constant_iterator.h>

namespace cudf {

// Applies a fixed-size rolling window function to the values in a column, with default output
// specified
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  return detail::rolling_window(input,
                                default_outputs,
                                preceding_window,
                                following_window,
                                min_periods,
                                agg,
                                cudf::default_stream_value,
                                mr);
}

// Applies a fixed-size rolling window function to the values in a column, without default specified
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  auto defaults =
    cudf::is_dictionary(input.type()) ? dictionary_column_view(input).indices() : input;
  return rolling_window(
    input, empty_like(defaults)->view(), preceding_window, following_window, min_periods, agg, mr);
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  return detail::rolling_window(
    input, preceding_window, following_window, min_periods, agg, cudf::default_stream_value, mr);
}

}  // namespace cudf
