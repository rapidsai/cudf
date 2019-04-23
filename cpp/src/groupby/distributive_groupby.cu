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
#include "new_hash_groupby.hpp"

#include <vector>

namespace cudf {
namespace groupby {

std::tuple<cudf::table, cudf::table> distributive(
    cudf::table const& keys, cudf::table const& values,
    std::vector<distributive_operators> const& operators,
    std::vector<gdf_dtype> const& output_dtypes) {
  CUDF_EXPECTS(
      static_cast<gdf_size_type>(operators.size()) == values.num_columns(),
      "Size mismatch between operators and value columns");

  for (gdf_size_type i = 0; i < values.num_columns(); ++i) {
    if ((operators[i] == SUM) and
        (values.get_column(i)->dtype == GDF_STRING_CATEGORY)) {
      CUDF_FAIL(
          "Cannot compute SUM aggregation of GDF_STRING_CATEGORY column.");
    }
  }

  return cudf::detail::hash_groupby(keys, values, operators, output_dtypes);
}
}  // namespace groupby
}  // namespace cudf