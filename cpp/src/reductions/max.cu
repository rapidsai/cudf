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
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <reductions/simple.cuh>

namespace cudf {
namespace reduction {

namespace {

template <typename Op>
struct max_fn_dispatcher {
 private:
  template <typename ElementType>
  static constexpr bool is_supported_v()
  {
    return !(cudf::is_fixed_point<ElementType>() ||
             std::is_same<ElementType, cudf::list_view>::value ||
             std::is_same<ElementType, cudf::struct_view>::value);
  }

 public:
  template <
    typename ElementType,
    std::enable_if_t<is_supported_v<ElementType>() and cudf::is_numeric<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    if (cudf::is_dictionary(col.type())) {
      auto indices = cudf::dictionary_column_view(col).get_indices_annotated();
      auto key_index_scalar =
        cudf::reduction::simple::simple_reduction<ElementType, ElementType, Op>(
          indices, indices.type(), mr, stream);
      auto key_index = static_cast<numeric_scalar<ElementType>*>(key_index_scalar.get());
      return cudf::detail::get_element(
        cudf::dictionary_column_view(col).keys(), key_index->value(stream), stream, mr);
    }
    return cudf::reduction::simple::simple_reduction<ElementType, ElementType, Op>(
      col, cudf::data_type{cudf::type_to_id<ElementType>()}, mr, stream);
  }

  template <
    typename ElementType,
    std::enable_if_t<is_supported_v<ElementType>() and !cudf::is_numeric<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    return cudf::reduction::simple::simple_reduction<ElementType, ElementType, Op>(
      col, cudf::data_type{cudf::type_to_id<ElementType>()}, mr, stream);
  }

  template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    CUDF_FAIL("Reduction operator `max` not supported for this type");
  }
};

}  // namespace

std::unique_ptr<cudf::scalar> max(column_view const& col,
                                  cudf::data_type const output_dtype,
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
{
  // using reducer = cudf::reduction::simple::element_type_dispatcher<cudf::reduction::op::max>;
  using reducer = max_fn_dispatcher<cudf::reduction::op::max>;

  auto col_type =
    cudf::is_dictionary(col.type()) ? dictionary_column_view(col).indices().type() : col.type();
  // return cudf::type_dispatcher(col_type, reducer(), col, output_dtype, mr, stream);

  auto result = cudf::type_dispatcher(col_type, reducer(), col, mr, stream);

  if (output_dtype == result->type() || !result->is_valid(stream)) return result;

  // if the output_dtype does not match, do extra work to cast it here
  auto input = cudf::make_column_from_scalar(*result, 1, mr, stream);
  // TODO: should build a scalar cast function
  auto output = cudf::detail::cast(*input, output_dtype, mr, stream);
  return cudf::detail::get_element(*output, 0, stream, mr);
}

}  // namespace reduction
}  // namespace cudf
