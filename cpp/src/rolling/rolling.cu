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

// See include/cudf/rolling.hpp for declaration
std::unique_ptr<column> rolling_window(column_view const &input,
                                       size_type window,
                                       size_type min_periods,
                                       size_type forward_window,
                                       rolling_operator agg_type)
{
  if (input.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);

  return cudf::make_numeric_column(data_type{INT32}, 0);
}

// See include/cudf/rolling.hpp for declaration
std::unique_ptr<column> rolling_window(column_view const &input,
                                       size_type window,
                                       size_type min_periods,
                                       size_type forward_window,
                                       std::string const& user_defined_aggregator,
                                       rolling_operator agg_op,
                                       data_type output_type)
{
  if (input.size() == 0)
    return cudf::make_numeric_column(data_type{INT32}, 0);

  return cudf::make_numeric_column(data_type{INT32}, 0);
}

} // namespace experimental 
} // namespace cudf
