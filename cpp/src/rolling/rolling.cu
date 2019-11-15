/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/rolling.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>

#include <memory>

namespace cudf {
namespace experimental {

namespace detail {

// Applies a rolling window function to the values in a column.
template <typename WindowIterator>
std::unique_ptr<column> rolling_window(column_view const& input,
                                       WindowIterator window_begin,
                                       WindowIterator window_end,
                                       WindowIterator forward_window_begin,
                                       size_type min_periods,
                                       rolling_operator op,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream = 0)
{
  return cudf::make_numeric_column(data_type{INT32}, 0);
}

// Applies a user-defined rolling window function to the values in a column.
template <typename WindowIterator>
std::unique_ptr<column> rolling_window(column_view const &input,
                                       WindowIterator window_begin,
                                       WindowIterator window_end,
                                       WindowIterator forward_window_begin,
                                       size_type min_periods,
                                       std::string const& user_defined_aggregator,
                                       rolling_operator agg_op,
                                       data_type output_type,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream = 0)
{
  return cudf::make_numeric_column(data_type{INT32}, 0);
}

} // namespace detail

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type window,
                                       size_type forward_window,
                                       size_type min_periods,
                                       rolling_operator op,
                                       rmm::mr::device_memory_resource* mr)
{
  auto window_begin = thrust::make_constant_iterator(window);
  auto window_end = thrust::make_constant_iterator(window);
  auto forward_window_begin = thrust::make_constant_iterator(forward_window);

  return cudf::experimental::detail::rolling_window(input, window_begin, window_end,
                                                    forward_window_begin, min_periods, op, mr, 0);
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& window,
                                       column_view const& forward_window,
                                       size_type min_periods,
                                       rolling_operator op,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.size() == 0 || window.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);
  
  CUDF_EXPECTS(window.type().id() == INT32 && forward_window.type().id() == INT32,
               "window/forward_window must have INT32 type");

  CUDF_EXPECTS(window.size() != input.size() && forward_window.size() != input.size(),
               "window/forward_window size must match input size");

  return cudf::experimental::detail::rolling_window(input, window.begin<size_type>(),
                                                    window.end<size_type>(),
                                                    forward_window.begin<size_type>(), min_periods,
                                                    op, mr, 0);
}

// Applies a fixed-size user-defined rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const &input,
                                       size_type window,
                                       size_type forward_window,
                                       size_type min_periods,
                                       std::string const& user_defined_aggregator,
                                       rolling_operator op,
                                       data_type output_type,
                                       rmm::mr::device_memory_resource* mr)
{
  auto window_begin = thrust::make_constant_iterator(window);
  auto window_end = thrust::make_constant_iterator(window);
  auto forward_window_begin = thrust::make_constant_iterator(forward_window);

  return cudf::experimental::detail::rolling_window(input, window_begin, window_end, 
                                                    forward_window_begin, min_periods,
                                                    user_defined_aggregator, op, output_type, mr, 0);
}

// Applies a variable-size user-defined rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const &input,
                                       column_view const& window,
                                       column_view const& forward_window,
                                       size_type min_periods,
                                       std::string const& user_defined_aggregator,
                                       rolling_operator op,
                                       data_type output_type,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.size() == 0 || window.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);
  
  CUDF_EXPECTS(window.type().id() == INT32 && forward_window.type().id() == INT32,
               "window/forward_window must have INT32 type");

  CUDF_EXPECTS(window.size() != input.size() && forward_window.size() != input.size(),
               "window/forward_window size must match input size");

  return cudf::experimental::detail::rolling_window(input, window.begin<size_type>(),
                                                    window.end<size_type>(),
                                                    forward_window.begin<size_type>(), min_periods,
                                                    user_defined_aggregator, op, output_type, mr, 0);
}

} // namespace experimental 
} // namespace cudf
