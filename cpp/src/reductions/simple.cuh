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

#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf {
namespace experimental {
namespace reduction {
namespace simple {


/** --------------------------------------------------------------------------*    
 * @brief Reduction for 'sum', 'product', 'min', 'max', 'sum of squares'
 * which directly compute the reduction by a single step reduction call
 *
 * @param[in] col    input column view
 * @param[in] mr The resource to use for all allocations
 * @param[in] stream cuda stream
 * @returns   Output scalar in device memory
 *
 * @tparam ElementType  the input column cudf dtype
 * @tparam ResultType   the output cudf dtype
 * @tparam Op           the operator of cudf::experimental::reduction::op::
 * ----------------------------------------------------------------------------**/
template <typename ElementType, typename ResultType, typename Op>
std::unique_ptr<scalar> simple_reduction(column_view const& col,
                                         data_type const output_dtype,
                                         rmm::mr::device_memory_resource* mr,
                                         cudaStream_t stream)
{
  // reduction by iterator
  auto dcol = cudf::column_device_view::create(col, stream);
  std::unique_ptr<scalar> result;
  Op simple_op{};

  if (col.has_nulls()) {
    auto it = thrust::make_transform_iterator(
      experimental::detail::make_null_replacement_iterator(*dcol, simple_op.template get_identity<ElementType>()),
      simple_op.template get_element_transformer<ResultType>());
    result = detail::reduce(it, col.size(), Op{}, mr, stream);
  } else {
    auto it = thrust::make_transform_iterator(
        dcol->begin<ElementType>(), 
        simple_op.template get_element_transformer<ResultType>());
    result = detail::reduce(it, col.size(), Op{}, mr, stream);
  }
  // set scalar is valid
  result->set_valid((col.null_count() < col.size()), stream);
  return result;
};

// @brief result type dispatcher for simple reduction (a.k.a. sum, prod, min...)
template <typename ElementType, typename Op>
struct result_type_dispatcher {
private:
    template <typename ResultType>
    static constexpr bool is_supported_v()
    {
      // for single step reductions,
      // the available combination of input and output dtypes are
      //  - same dtypes (including cudf::wrappers)
      //  - any arithmetic dtype to any arithmetic dtype
      //  - cudf::experimental::bool8 to/from any arithmetic dtype
      return std::is_convertible<ElementType, ResultType>::value &&
             (std::is_arithmetic<ResultType>::value ||
              std::is_same<Op, cudf::experimental::reduction::op::min>::value ||
              std::is_same<Op, cudf::experimental::reduction::op::max>::value);
    }

public:
    template <typename ResultType, std::enable_if_t<is_supported_v<ResultType>()>* = nullptr>
    std::unique_ptr<scalar> operator()(column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
    {
      return simple_reduction<ElementType, ResultType, Op>(col, output_dtype, mr, stream);
    }

    template <typename ResultType, std::enable_if_t<not is_supported_v<ResultType>()>* = nullptr>
    std::unique_ptr<scalar> operator()(column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
    {
        CUDF_FAIL("input data type is not convertible to output data type");
    }
};

// @brief input column element for simple reduction (a.k.a. sum, prod, min...)
template <typename Op>
struct element_type_dispatcher {
private:
    // return true if ElementType is arithmetic type or bool8, or
    // Op is DeviceMin or DeviceMax for wrapper (non-arithmetic) types
    template <typename ElementType>
    static constexpr bool is_supported_v()
    {
      // disable only for string ElementType except for operators min, max
      return  !( std::is_same<ElementType, cudf::string_view>::value &&
              !( std::is_same<Op, cudf::experimental::reduction::op::min>::value ||
                 std::is_same<Op, cudf::experimental::reduction::op::max>::value ));
    }

public:
    template <typename ElementType, std::enable_if_t<is_supported_v<ElementType>()>* = nullptr>
    std::unique_ptr<scalar> operator()(column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
    {
        return cudf::experimental::type_dispatcher(output_dtype,
            result_type_dispatcher<ElementType, Op>(), col, output_dtype, mr, stream);
    }

    template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
    std::unique_ptr<scalar> operator()(column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
    {
        CUDF_FAIL("Reduction operators other than `min` and `max`"
                  " are not supported for non-arithmetic types");
    }
};

} // namespace simple
} // namespace reduction
} // namespace experimental
} // namespace cudf

