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

#pragma once

#include <cudf/detail/reduction.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace reduction {
namespace simple {
/**
 * @brief Reduction for 'sum', 'product', 'min', 'max', 'sum of squares'
 * which directly compute the reduction by a single step reduction call
 *
 * @tparam ElementType  the input column data-type
 * @tparam ResultType   the output data-type
 * @tparam Op           the operator of cudf::reduction::op::

 * @param col Input column of data to reduce
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @param stream Used for device memory operations and kernel launches.
 * @return Output scalar in device memory
 */
template <typename ElementType, typename ResultType, typename Op>
std::unique_ptr<scalar> simple_reduction(column_view const& col,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  // reduction by iterator
  auto dcol = cudf::column_device_view::create(col, stream);
  std::unique_ptr<scalar> result;
  Op simple_op{};

  if (col.has_nulls()) {
    auto it = thrust::make_transform_iterator(
      dcol->pair_begin<ElementType, true>(),
      simple_op.template get_null_replacing_element_transformer<ResultType>());
    result = detail::reduce(it, col.size(), Op{}, stream, mr);
  } else {
    auto it = thrust::make_transform_iterator(
      dcol->begin<ElementType>(), simple_op.template get_element_transformer<ResultType>());
    result = detail::reduce(it, col.size(), Op{}, stream, mr);
  }
  // set scalar is valid
  result->set_valid((col.null_count() < col.size()), stream);
  return result;
};

/**
 * @brief Convert a numeric scalar to another numeric scalar.
 *
 * The input value and validity are cast to the output scalar.
 *
 * @tparam InputType The type of the input scalar to copy from
 * @tparam OutputType The output scalar type to copy to
 */
template <typename InputType, typename OutputType>
struct assign_scalar_fn {
  __device__ void operator()()
  {
    d_output.set_value(static_cast<OutputType>(d_input.value()));
    d_output.set_valid(d_input.is_valid());
  }

  cudf::numeric_scalar_device_view<InputType> d_input;
  cudf::numeric_scalar_device_view<OutputType> d_output;
};

/**
 * @brief A type-dispatcher functor for converting a numeric scalar.
 *
 * The InputType is known and the dispatch is on the ResultType
 * which is the output numeric scalar type.
 *
 * @tparam InputType The scalar type to convert from
 */
template <typename InputType>
struct cast_numeric_scalar_fn {
 private:
  template <typename ResultType>
  static constexpr bool is_supported_v()
  {
    return cudf::is_convertible<InputType, ResultType>::value && cudf::is_numeric<ResultType>();
  }

 public:
  template <typename ResultType, std::enable_if_t<is_supported_v<ResultType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(numeric_scalar<InputType>* input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto d_input  = cudf::get_scalar_device_view(*input);
    auto result   = std::make_unique<numeric_scalar<ResultType>>(ResultType{}, true, stream, mr);
    auto d_output = cudf::get_scalar_device_view(*result);
    cudf::detail::device_single_thread(assign_scalar_fn<InputType, ResultType>{d_input, d_output},
                                       stream);
    return result;
  }

  template <typename ResultType, std::enable_if_t<not is_supported_v<ResultType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(numeric_scalar<InputType>* input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("input data type is not convertible to output data type");
  }
};

/**
 * @brief Call reduce and return a scalar of type bool.
 *
 * This is used by operations any() and all().
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct bool_result_element_dispatcher {
 private:
  template <typename ElementType>
  static constexpr bool is_supported_v()
  {
    return std::is_arithmetic<ElementType>();
  }

 public:
  template <typename ElementType, std::enable_if_t<is_supported_v<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return cudf::reduction::simple::simple_reduction<ElementType, bool, Op>(col, stream, mr);
  }

  template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

/**
 * @brief Call reduce and return a scalar of type matching the input column.
 *
 * This is used by operations min() and max().
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct same_element_type_dispatcher {
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
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return cudf::reduction::simple::simple_reduction<ElementType, ElementType, Op>(col, stream, mr);
  }

  template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Reduction operator `max` not supported for this type");
  }
};

/**
 * @brief Call reduce and return a scalar of the type specified.
 *
 * This is used by operations sum(), product(), and sum_of_squares().
 * It only supports numeric types. If the output type is not the
 * same as the input type, an extra cast operation may incur.
 *
 * @tparam Op The reduce operation to execute on the column.
 */
template <typename Op>
struct element_type_dispatcher {
  template <typename ElementType,
            typename std::enable_if_t<std::is_floating_point<ElementType>::value>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     data_type const output_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    if (output_type == col.type())
      return cudf::reduction::simple::simple_reduction<ElementType, ElementType, Op>(
        col, stream, mr);
    auto result =
      cudf::reduction::simple::simple_reduction<ElementType, double, Op>(col, stream, mr);
    if (output_type == result->type()) return result;
    // this will cast the result to the output_type
    return cudf::type_dispatcher(output_type,
                                 simple::cast_numeric_scalar_fn<double>{},
                                 static_cast<numeric_scalar<double>*>(result.get()),
                                 stream,
                                 mr);
  }

  template <typename ElementType,
            typename std::enable_if_t<std::is_integral<ElementType>::value>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     data_type const output_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    if (output_type == col.type())
      return cudf::reduction::simple::simple_reduction<ElementType, ElementType, Op>(
        col, stream, mr);
    auto result =
      cudf::reduction::simple::simple_reduction<ElementType, int64_t, Op>(col, stream, mr);
    if (output_type == result->type()) return result;
    // this will cast the result to the output_type
    return cudf::type_dispatcher(output_type,
                                 simple::cast_numeric_scalar_fn<int64_t>{},
                                 static_cast<numeric_scalar<int64_t>*>(result.get()),
                                 stream,
                                 mr);
    return result;
  }

  template <typename ElementType,
            typename std::enable_if_t<!std::is_floating_point<ElementType>::value and
                                      !std::is_integral<ElementType>::value>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     data_type const output_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Reduction operator not supported for this type");
  }
};

}  // namespace simple
}  // namespace reduction
}  // namespace cudf
