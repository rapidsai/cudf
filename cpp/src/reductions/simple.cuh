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

#include "reduction.cuh"
#include "reduction_operators.cuh"

#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf {
namespace experimental {
namespace reduction {
namespace simple {

// @brief identity specialized for string_view 
// because string_view constructor requires 2 arguments
template <typename T, typename Op,
          typename std::enable_if<!std::is_same<string_view, T>::value>::type* = nullptr>
constexpr T string_supported_identity() { return Op::Op::template identity<T>(); }
template <typename T, typename Op,
          typename std::enable_if< std::is_same<string_view, T>::value>::type* = nullptr>
constexpr T string_supported_identity() { return T{nullptr, 0}; }

/** --------------------------------------------------------------------------*    
 * @brief Reduction for 'sum', 'product', 'min', 'max', 'sum of squares'
 * which directly compute the reduction by a single step reduction call
 *
 * @param[in] col    input column view
 * @params[in] mr The resource to use for all allocations
 * @param[in] stream cuda stream
 * @returns unique_ptr<scalar>  output scalar data
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
  ResultType identity =  Op::Op::template identity<ResultType>();
  //  string_supported_identity<ResultType, Op>();
  rmm::device_scalar<ResultType> dev_result{identity, stream, mr}; 

  // reduction by iterator
  auto dcol = cudf::column_device_view::create(col, stream);
  if (col.has_nulls()) {
    auto it = thrust::make_transform_iterator(
        experimental::detail::make_null_replacement_iterator(
            *dcol, Op::Op::template identity<ElementType>()),
        typename Op::template transformer<ResultType>{});
    detail::reduce(dev_result.data(), it, col.size(), identity,
                   typename Op::Op{}, mr, stream);
  } else {
    auto it = thrust::make_transform_iterator(
        dcol->begin<ElementType>(),
        typename Op::template transformer<ResultType>{});
    detail::reduce(dev_result.data(), it, col.size(), identity,
                   typename Op::Op{}, mr, stream);
  }

  using ScalarType = cudf::experimental::scalar_type_t<ResultType>;
  auto s = new ScalarType(dev_result.value(), true, stream, mr);
  std::unique_ptr<scalar> result = std::unique_ptr<scalar>(s);

  // set scalar is valid
  if (col.null_count() < col.size())
    result->set_valid(true, stream);
  else
    result->set_valid(false, stream);
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

