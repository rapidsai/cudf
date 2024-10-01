/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/reduction/detail/reduction.cuh>
#include <cudf/reduction/detail/reduction_operators.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <stdexcept>
#include <type_traits>

namespace cudf {
namespace reduction {
namespace compound {
namespace detail {
/**
 * @brief Multi-step reduction for operations such as mean, variance, and standard deviation.
 *
 * @tparam ElementType  the input column data-type
 * @tparam ResultType   the output data-type
 * @tparam Op           the compound operator derived from `cudf::reduction::op::compound_op`
 *
 * @param col input column view
 * @param output_dtype data type of return type and typecast elements of input column
 * @param ddof Delta degrees of freedom used for standard deviation and variance. The divisor used
 * is N - ddof, where N represents the number of elements.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Output scalar in device memory
 */
template <typename ElementType, typename ResultType, typename Op>
std::unique_ptr<scalar> compound_reduction(column_view const& col,
                                           data_type const output_dtype,
                                           size_type ddof,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const valid_count = col.size() - col.null_count();

  // All null input produces all null output
  if (valid_count == 0 ||
      // Only care about ddof for standard deviation and variance right now
      valid_count <= ddof && (std::is_same_v<Op, cudf::reduction::detail::op::standard_deviation> ||
                              std::is_same_v<Op, cudf::reduction::detail::op::variance>)) {
    auto result = cudf::make_fixed_width_scalar(output_dtype, stream, mr);
    result->set_valid_async(false, stream);
    return result;
  }
  // reduction by iterator
  auto dcol = cudf::column_device_view::create(col, stream);
  Op compound_op{};

  if (!cudf::is_dictionary(col.type())) {
    if (col.has_nulls()) {
      auto it = thrust::make_transform_iterator(
        dcol->pair_begin<ElementType, true>(),
        compound_op.template get_null_replacing_element_transformer<ResultType>());
      return cudf::reduction::detail::reduce<Op, decltype(it), ResultType>(
        it, col.size(), compound_op, valid_count, ddof, stream, mr);
    } else {
      auto it = thrust::make_transform_iterator(
        dcol->begin<ElementType>(), compound_op.template get_element_transformer<ResultType>());
      return cudf::reduction::detail::reduce<Op, decltype(it), ResultType>(
        it, col.size(), compound_op, valid_count, ddof, stream, mr);
    }
  } else {
    auto it = thrust::make_transform_iterator(
      cudf::dictionary::detail::make_dictionary_pair_iterator<ElementType>(*dcol, col.has_nulls()),
      compound_op.template get_null_replacing_element_transformer<ResultType>());
    return cudf::reduction::detail::reduce<Op, decltype(it), ResultType>(
      it, col.size(), compound_op, valid_count, ddof, stream, mr);
  }
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
    return std::is_floating_point_v<ResultType>;
  }

 public:
  template <typename ResultType, std::enable_if_t<is_supported_v<ResultType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     cudf::data_type const output_dtype,
                                     size_type ddof,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    return compound_reduction<ElementType, ResultType, Op>(col, output_dtype, ddof, stream, mr);
  }

  template <typename ResultType, std::enable_if_t<not is_supported_v<ResultType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     cudf::data_type const output_dtype,
                                     size_type ddof,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
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
    return std::is_arithmetic_v<ElementType>;
  }

 public:
  template <typename ElementType, std::enable_if_t<is_supported_v<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     cudf::data_type const output_dtype,
                                     size_type ddof,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    CUDF_EXPECTS(ddof >= 0, "ddof must be non-negative", std::domain_error);
    return cudf::type_dispatcher(
      output_dtype, result_type_dispatcher<ElementType, Op>(), col, output_dtype, ddof, stream, mr);
  }

  template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     cudf::data_type const output_dtype,
                                     size_type ddof,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    CUDF_FAIL(
      "Reduction operators other than `min` and `max`"
      " are not supported for non-arithmetic types");
  }
};

}  // namespace detail
}  // namespace compound
}  // namespace reduction
}  // namespace cudf
