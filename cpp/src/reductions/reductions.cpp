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

#include <cudf/reduction.hpp>
#include "reduction_functions.hpp"

namespace cudf {
namespace experimental {

gdf_scalar reduce(column_view const& col,
                  cudf::experimental::reduction::operators op,
                  cudf::data_type output_dtype, cudf::size_type ddof)
{
  gdf_scalar scalar;
  //scalar.dtype = output_dtype; //TODO after cudf::scalar support
  scalar.is_valid = false;  // the scalar is not valid for error case

  //TODO CUDF_EXPECTS(col != nullptr, "Input column is null");
  // check if input column is empty
  if (col.size() <= col.null_count()) return scalar;

  switch (op) {
    case cudf::experimental::reduction::SUM:
      scalar = cudf::experimental::reduction::sum(col, output_dtype);
      break;
    case cudf::experimental::reduction::MIN:
      scalar = cudf::experimental::reduction::min(col, output_dtype);
      break;
    case cudf::experimental::reduction::MAX:
      scalar = cudf::experimental::reduction::max(col, output_dtype);
      break;
    case cudf::experimental::reduction::ANY:
      scalar = cudf::experimental::reduction::any(col, output_dtype);
      break;
    case cudf::experimental::reduction::ALL:
      scalar = cudf::experimental::reduction::all(col, output_dtype);
      break;
    case cudf::experimental::reduction::PRODUCT:
      scalar = cudf::experimental::reduction::product(col, output_dtype);
      break;
    case cudf::experimental::reduction::SUMOFSQUARES:
      scalar =
          cudf::experimental::reduction::sum_of_squares(col, output_dtype);
      break;

    case cudf::experimental::reduction::MEAN:
      scalar = cudf::experimental::reduction::mean(col, output_dtype);
      break;
    case cudf::experimental::reduction::VAR:
      scalar = cudf::experimental::reduction::variance(col, output_dtype, ddof);
      break;
    case cudf::experimental::reduction::STD:
      scalar = cudf::experimental::reduction::standard_deviation(col, output_dtype, ddof);
      break;
    default:
      CUDF_FAIL("Unsupported reduction operator");
  }

  return scalar;
}

}  // namespace experimental
}  // namespace cudf

