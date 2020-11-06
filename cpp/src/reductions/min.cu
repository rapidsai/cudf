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

// TODO: This is the same code as max_fn_dispatcher -- move to simple.cuh?
template <typename Op>
struct min_fn_dispatcher {
 private:
  template <typename ElementType>
  static constexpr bool is_supported_v()
  {
    return !(cudf::is_fixed_point<ElementType>() ||
             std::is_same<ElementType, cudf::list_view>::value ||
             std::is_same<ElementType, cudf::struct_view>::value);
  }

 public:
  template <typename ElementType, std::enable_if_t<is_supported_v<ElementType>()>* = nullptr>
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
    CUDF_FAIL("Reduction operator `min` not supported for this type");
  }
};

}  // namespace

std::unique_ptr<cudf::scalar> min(column_view const& col,
                                  data_type const output_dtype,
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
{
  using reducer = min_fn_dispatcher<cudf::reduction::op::min>;

  auto col_type =
    cudf::is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();

  std::unique_ptr<cudf::scalar> result;
  if (!cudf::is_dictionary(col.type())) {
    result = cudf::type_dispatcher(col_type, reducer(), col, mr, stream);
  } else {
    cudf::reduction::op::min simple_op{};
    // TODO: Need a pair indexalator
    // if (col.has_nulls()) {
    //  auto it = thrust::make_transform_iterator(
    //    cudf::dictionary::detail::make_dictionary_pair_iterator<ElementType, true>(*dcol),
    //    simple_op.template get_null_replacing_element_transformer<ResultType>());
    //  result = detail::reduce(it, col.size(), Op{}, mr, stream);
    //} else {
    auto dict_col = cudf::dictionary_column_view(col);
    auto indices  = dict_col.get_indices_annotated();
    auto it       =  // thrust::make_transform_iterator(
      cudf::detail::indexalator_factory::make_input_iterator(indices);  //,
    // simple_op.get_element_transformer<size_type>());
    result         = cudf::reduction::detail::reduce(it, col.size(), simple_op, mr, stream);
    auto key_index = static_cast<numeric_scalar<size_type>*>(result.get());
    result = cudf::detail::get_element(dict_col.keys(), key_index->value(stream), stream, mr);
    //}
  }

  if (output_dtype == col_type) return result;

  // if the output_dtype does not match, do extra work to cast it here
  auto input = cudf::make_column_from_scalar(*result, 1, mr, stream);
  // TODO: should build a scalar cast function
  auto output = cudf::detail::cast(*input, output_dtype, mr, stream);
  return cudf::detail::get_element(*output, 0, stream, mr);
}

}  // namespace reduction
}  // namespace cudf
