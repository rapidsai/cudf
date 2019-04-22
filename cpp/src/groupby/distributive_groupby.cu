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

#include <groupby.hpp>
#include <types.hpp>
#include <utilities/error_utils.hpp>

#include <vector>

namespace cudf {
namespace groupby {

std::tuple<cudf::table, cudf::table> distributive(
    cudf::table const& keys, cudf::table const& values,
    std::vector<distributive_operators> const& operators) {

  CUDF_EXPECTS(
      static_cast<gdf_size_type>(operators.size()) == values.num_columns(),
      "Size mismatch between operators and value columns");

  return std::make_tuple(keys, values);
}
}  // namespace groupby
}  // namespace cudf