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
#include <cudf/scalar/scalar_factories.hpp>
#include "reduction_functions.hpp"

namespace cudf {
namespace experimental {
namespace detail {

// Allocate storage for a single identity scalar
std::unique_ptr<scalar> make_identity_scalar(
    data_type type, cudaStream_t stream=0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
  if(is_numeric(type))
    return  make_numeric_scalar(type, stream, mr);
  else if(is_timestamp(type))
    return make_timestamp_scalar(type, stream, mr);
  else if(type == data_type(STRING))
    return make_string_scalar("", stream, mr);
  else 
    CUDF_FAIL("Invalid type.");
}

std::unique_ptr<scalar> reduce(
    column_view const& col, reduction::operators op, data_type output_dtype,
    size_type ddof, cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  std::unique_ptr<scalar> result = make_identity_scalar(output_dtype, stream, mr);

  // check if input column is empty
  if (col.size() <= col.null_count()) return result;

  switch (op) {
    case reduction::SUM:
      result = reduction::sum(col, output_dtype, stream, mr);
      break;
    case reduction::MIN:
      result = reduction::min(col, output_dtype, stream, mr);
      break;
    case reduction::MAX:
      result = reduction::max(col, output_dtype, stream, mr);
      break;
    case reduction::ANY:
      result = reduction::any(col, output_dtype, stream, mr);
      break;
    case reduction::ALL:
      result = reduction::all(col, output_dtype, stream, mr);
      break;
    case reduction::PRODUCT:
      result = reduction::product(col, output_dtype, stream, mr);
      break;
    case reduction::SUMOFSQUARES:
      result =
          reduction::sum_of_squares(col, output_dtype, stream, mr);
      break;

    case reduction::MEAN:
      result = reduction::mean(col, output_dtype, stream, mr);
      break;
    case reduction::VAR:
      result = reduction::variance(col, output_dtype, ddof, stream, mr);
      break;
    case reduction::STD:
      result = reduction::standard_deviation(col, output_dtype, ddof, stream, mr);
      break;
    default:
      CUDF_FAIL("Unsupported reduction operator");
  }

  return result;
}
}  // namespace detail

 std::unique_ptr<scalar> reduce(
    column_view const& col, reduction::operators op, data_type output_dtype,
    size_type ddof,
    rmm::mr::device_memory_resource* mr)
{
  return detail::reduce(col, op, output_dtype, ddof, 0, mr);
}

}  // namespace experimental
}  // namespace cudf

