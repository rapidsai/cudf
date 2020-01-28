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

#pragma once 

#include <cudf/detail/reduction.cuh>

#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <rmm/device_scalar.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>

namespace cudf {
namespace experimental {
namespace reduction {
namespace compound {

/** --------------------------------------------------------------------------*
 * @brief Multi-step reduction for operations such as mean and variance, and
 * standard deviation.
 *
 * @param[in] col    input column view
 * @param[in] ddof   `Delta Degrees of Freedom` used for `std`, `var`.
 *                   The divisor used in calculations is N - ddof, where N
 *                   represents the number of elements.
 * @param[in] mr    The resource to use for all allocations
 * @param[in] stream cuda stream
 * @returns   Output scalar in device memory
 *
 * @tparam ElementType  the input column cudf dtype
 * @tparam ResultType   the output cudf dtype
 * @tparam Op           the compound operator derived from `cudf::experimental::reduction::op::compound_op`
 * ----------------------------------------------------------------------------**/
template <typename ElementType, typename ResultType, typename Op>
std::unique_ptr<scalar> compound_reduction(column_view const& col,
                                           cudf::size_type ddof,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
{
  cudf::size_type valid_count = col.size() - col.null_count();

  // reduction by iterator
  auto dcol = cudf::column_device_view::create(col, stream);
  std::unique_ptr<scalar> result;
  Op compound_op{};
  
  if (col.has_nulls()) {
    auto it = thrust::make_transform_iterator(
      experimental::detail::make_null_replacement_iterator(*dcol, compound_op.template get_identity<ElementType>()),
      compound_op.template get_element_transformer<ResultType>());
    result = detail::reduce<Op, decltype(it), ResultType>(it, col.size(), compound_op, valid_count, ddof, mr, stream);
  } else {
    auto it = thrust::make_transform_iterator(
        dcol->begin<ElementType>(), 
        compound_op.template get_element_transformer<ResultType>());
    result = detail::reduce<Op, decltype(it), ResultType>(it, col.size(), compound_op, valid_count, ddof, mr, stream);
  }
  // set scalar is valid
  result->set_valid((col.null_count() < col.size()), stream);
  return result;
};

// @brief input column element dispatcher for compound reduction (a.k.a. mean, var, std)
template <typename Op, cudf::experimental::aggregation::Kind k>
struct element_type_dispatcher {
private:
    // return true if ElementType is arithmetic type
  template <typename ElementType> static constexpr bool is_supported_v()
  {
    using ResultType = cudf::experimental::detail::target_type_t<ElementType, k>;
    return std::is_convertible<ElementType, ResultType>::value &&
           std::is_arithmetic<ElementType>::value;
  }

public:
    template <typename ElementType, std::enable_if_t<is_supported_v<ElementType>()>* = nullptr>
    std::unique_ptr<scalar> operator()(column_view const& col, cudf::size_type ddof, 
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
    {
      using ResultType = cudf::experimental::detail::target_type_t<ElementType, k>;
      return compound_reduction<ElementType, ResultType, Op>(col, ddof, mr, stream);
    }

    template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
    std::unique_ptr<scalar> operator()(column_view const& col, cudf::size_type ddof,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
    {
        CUDF_FAIL("Reduction operators other than `min` and `max`"
                  " are not supported for non-arithmetic types");
    }
};

} // namespace compound
} // namespace reduction
} // namespace experimental
} // namespace cudf

