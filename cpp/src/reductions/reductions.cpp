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
#include <cudf/detail/reduction_functions.hpp>

namespace cudf {
namespace experimental {
namespace detail {

std::unique_ptr<scalar> reduce(
    column_view const& col, reduction_op op, data_type output_dtype,
    size_type ddof, 
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0)
{
  std::unique_ptr<scalar> result = make_default_constructed_scalar(output_dtype);
  result->set_valid(false, stream);

  // check if input column is empty
  if (col.size() <= col.null_count()) return result;

  switch (op) {
    case reduction_op::SUM:
      result = reduction::sum(col, output_dtype, mr, stream);
      break;
    case reduction_op::MIN:
      result = reduction::min(col, output_dtype, mr, stream);
      break;
    case reduction_op::MAX:
      result = reduction::max(col, output_dtype, mr, stream);
      break;
    case reduction_op::ANY:
      result = reduction::any(col, output_dtype, mr, stream);
      break;
    case reduction_op::ALL:
      result = reduction::all(col, output_dtype, mr, stream);
      break;
    case reduction_op::PRODUCT:
      result = reduction::product(col, output_dtype, mr, stream);
      break;
    case reduction_op::SUMOFSQUARES:
      result =
          reduction::sum_of_squares(col, output_dtype, mr, stream);
      break;

    case reduction_op::MEAN:
      result = reduction::mean(col, output_dtype, mr, stream);
      break;
    case reduction_op::VAR:
      result = reduction::variance(col, output_dtype, ddof, mr, stream);
      break;
    case reduction_op::STD:
      result = reduction::standard_deviation(col, output_dtype, ddof, mr, stream);
      break;
    default:
      CUDF_FAIL("Unsupported reduction operator");
  }

  return result;
}
}  // namespace detail

 std::unique_ptr<scalar> reduce(
    column_view const& col, reduction_op op, data_type output_dtype,
    size_type ddof,
    rmm::mr::device_memory_resource* mr)
{
  return detail::reduce(col, op, output_dtype, ddof, mr);
}

}  // namespace experimental
}  // namespace cudf

