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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <reductions/simple.cuh>

namespace cudf {
namespace reduction {
namespace {

// TODO: This may become the new result_type_dispatche
template <typename Op>
struct same_type_dispatcher {
  template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    if (cudf::is_dictionary(col.type()))
      return simple::dictionary_reduction<T, double, Op>(col, mr, stream);
    return cudf::reduction::simple::simple_reduction<T, double, Op>(
      col, cudf::data_type{cudf::type_to_id<double>()}, mr, stream);
  }

  template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    if (cudf::is_dictionary(col.type()))
      return simple::dictionary_reduction<T, int64_t, Op>(col, mr, stream);
    return cudf::reduction::simple::simple_reduction<T, int64_t, Op>(
      col, cudf::data_type{cudf::type_to_id<int64_t>()}, mr, stream);
  }

  template <typename T,
            typename std::enable_if_t<!std::is_floating_point<T>::value and
                                      !std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

}  // namespace

std::unique_ptr<cudf::scalar> sum_of_squares(column_view const& col,
                                             cudf::data_type const output_dtype,
                                             rmm::mr::device_memory_resource* mr,
                                             cudaStream_t stream)
{
  using reducer = same_type_dispatcher<cudf::reduction::op::sum_of_squares>;

  auto col_type =
    cudf::is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();

  auto result = cudf::type_dispatcher(col_type, reducer(), col, mr, stream);

  if (output_dtype == result->type() || !result->is_valid(stream)) return result;

  // if the output_dtype does not match, do extra work to cast it here
  auto input = cudf::make_column_from_scalar(*result, 1, mr, stream);
  // should build a scalar cast function
  auto output = cudf::detail::cast(*input, output_dtype, mr, stream);
  return cudf::detail::get_element(*output, 0, stream, mr);
}

}  // namespace reduction
}  // namespace cudf
