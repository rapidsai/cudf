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

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type window,
                                       size_type min_periods,
                                       size_type forward_window,
                                       rolling_operator op,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);

  return cudf::make_numeric_column(data_type{INT32}, 0);
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& window,
                                       size_type min_periods,
                                       size_type forward_window,
                                       rolling_operator op,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.size() == 0 || window.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);
  
  CUDF_EXPECTS(window.type().id() == INT32, "window must have INT32 type");

  return cudf::make_numeric_column(data_type{INT32}, 0);
}

// Applies a fixed-size user-defined rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const &input,
                                       size_type window,
                                       size_type min_periods,
                                       size_type forward_window,
                                       std::string const& user_defined_aggregator,
                                       rolling_operator agg_op,
                                       data_type output_type,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);

  return cudf::make_numeric_column(data_type{INT32}, 0);
}

// Applies a variable-size user-defined rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const &input,
                                       column_view const& window,
                                       size_type min_periods,
                                       size_type forward_window,
                                       std::string const& user_defined_aggregator,
                                       rolling_operator agg_op,
                                       data_type output_type,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.size() == 0 || window.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);
  
  CUDF_EXPECTS(window.type().id() == INT32, "window must have INT32 type");

  return cudf::make_numeric_column(data_type{INT32}, 0);
}

} // namespace experimental 
} // namespace cudf
