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
                                           data_type const output_dtype,
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
  if (col.null_count() < col.size())
    result->set_valid(true, stream);
  else
    result->set_valid(false, stream);
  return result;
};

// @brief result type dispatcher for compound reduction (a.k.a. mean, var, std)
template <typename ElementType, typename Op>
struct result_type_dispatcher {
private:
    template <typename ResultType>
    static constexpr bool is_supported_v()
    {
        // the operator `mean`, `var`, `std` only accepts
        // floating points as output dtype
        return  std::is_floating_point<ResultType>::value;
    }

public:
    template <typename ResultType, std::enable_if_t<is_supported_v<ResultType>()>* = nullptr>
    std::unique_ptr<scalar> operator()(column_view const& col, cudf::data_type const output_dtype, cudf::size_type ddof,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
    {
      return compound_reduction<ElementType, ResultType, Op>(col, output_dtype, ddof, mr, stream);
    }

    template <typename ResultType, std::enable_if_t<not is_supported_v<ResultType>()>* = nullptr >
    std::unique_ptr<scalar> operator()(column_view const& col, cudf::data_type const output_dtype, cudf::size_type ddof,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
    {
        CUDF_FAIL("Unsupported output data type");
    }
};

// @brief input column element dispatcher for compound reduction (a.k.a. mean, var, std)
template <typename Op>
struct element_type_dispatcher {
private:
    // return true if ElementType is arithmetic type 
    template <typename ElementType>
    static constexpr bool is_supported_v()
    {
        return std::is_arithmetic<ElementType>::value;
    }

public:
    template <typename ElementType, std::enable_if_t<is_supported_v<ElementType>()>* = nullptr>
    std::unique_ptr<scalar> operator()(column_view const& col, cudf::data_type const output_dtype, cudf::size_type ddof, 
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
    {
        return cudf::experimental::type_dispatcher(output_dtype,
            result_type_dispatcher<ElementType, Op>(), col, output_dtype, ddof, mr, stream);
    }

    template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
    std::unique_ptr<scalar> operator()(column_view const& col, cudf::data_type const output_dtype, cudf::size_type ddof,
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

