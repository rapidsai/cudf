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

#include <cudf/detail/reduction_functions.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <reductions/simple.cuh>

namespace cudf {
namespace reduction {
namespace {

template <typename Op>
struct any_fn_dispatcher {
 private:
  template <typename ElementType>
  static constexpr bool is_supported_v()
  {
    return std::is_arithmetic<ElementType>();
  }

 public:
  template <typename ElementType, std::enable_if_t<is_supported_v<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    // the dictionary indices can be used for reduce for this operation
    auto min_col = cudf::is_dictionary(col.type())
                     ? cudf::dictionary_column_view(col).get_indices_annotated()
                     : col;
    return cudf::reduction::simple::simple_reduction<ElementType, bool, Op>(
      min_col, cudf::data_type{cudf::type_id::BOOL8}, mr, stream);
  }

  template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    CUDF_FAIL("Reduction operator `any` not supported for this type");
  }
};

}  // namespace

std::unique_ptr<cudf::scalar> any(column_view const& col,
                                  cudf::data_type const output_dtype,
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
{
  CUDF_EXPECTS(output_dtype == cudf::data_type(cudf::type_id::BOOL8),
               "any() operation can be applied with output type `bool8` only");
  // return cudf::reduction::max(col, cudf::data_type(cudf::type_id::BOOL8), mr, stream);

  using reducer = any_fn_dispatcher<cudf::reduction::op::max>;
  auto col_type =  // we can use just the dictionary indices for this op
    cudf::is_dictionary(col.type()) ? dictionary_column_view(col).indices().type() : col.type();
  return cudf::type_dispatcher(col_type, reducer(), col, mr, stream);
}

}  // namespace reduction
}  // namespace cudf
