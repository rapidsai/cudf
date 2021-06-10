/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "binary_ops.hpp"
#include "operation.cuh"

#include <cudf/column/column_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace binops {
namespace compiled {

/**
 * @brief Type casts each element of the column to `CastType`
 *
 */
template <typename CastType>
struct type_casted_accessor {
  template <typename Element>
  CUDA_DEVICE_CALLABLE CastType operator()(cudf::size_type i,
                                           column_device_view const& col,
                                           bool is_scalar) const
  {
    if constexpr (column_device_view::has_element_accessor<Element>() and
                  std::is_convertible_v<Element, CastType>)
      return static_cast<CastType>(col.element<Element>(is_scalar ? 0 : i));
    return {};
  }
};

/**
 * @brief Type casts value to column type and stores in `i`th row of the column
 *
 */
template <typename FromType>
struct typed_casted_writer {
  template <typename Element>
  CUDA_DEVICE_CALLABLE void operator()(cudf::size_type i,
                                       mutable_column_device_view const& col,
                                       FromType val) const
  {
    if constexpr (mutable_column_device_view::has_element_accessor<Element>() and
                  std::is_constructible_v<Element, FromType>) {
      col.element<Element>(i) = static_cast<Element>(val);
    } else if constexpr (is_fixed_point<Element>() and std::is_constructible_v<Element, FromType>) {
      if constexpr (is_fixed_point<FromType>())
        col.data<Element::rep>()[i] = val.rescaled(numeric::scale_type{col.type().scale()}).value();
      else
        col.data<Element::rep>()[i] = Element{val, numeric::scale_type{col.type().scale()}}.value();
    }
  }
};

// Functors to launch only defined operations.

/**
 * @brief Functor to launch only defined operations with common type.
 *
 * @tparam BinaryOperator binary operator functor
 */
template <typename BinaryOperator, bool store_as_result = false>
struct ops_wrapper {
  mutable_column_device_view& out;
  column_device_view const& lhs;
  column_device_view const& rhs;
  bool is_lhs_scalar;
  bool is_rhs_scalar;
  template <typename TypeCommon>
  __device__ void operator()(size_type i)
  {
    if constexpr (std::is_invocable_v<BinaryOperator, TypeCommon, TypeCommon>) {
      TypeCommon x =
        type_dispatcher(lhs.type(), type_casted_accessor<TypeCommon>{}, i, lhs, is_lhs_scalar);
      TypeCommon y =
        type_dispatcher(rhs.type(), type_casted_accessor<TypeCommon>{}, i, rhs, is_rhs_scalar);
      auto result = [&]() {
        if constexpr (std::is_same_v<BinaryOperator, ops::NullEquals> or
                      std::is_same_v<BinaryOperator, ops::NullMax> or
                      std::is_same_v<BinaryOperator, ops::NullMin>) {
          bool output_valid = false;
          auto result       = BinaryOperator{}.template operator()<TypeCommon, TypeCommon>(
            x,
            y,
            lhs.is_valid(is_lhs_scalar ? 0 : i),
            rhs.is_valid(is_rhs_scalar ? 0 : i),
            output_valid);
          if (out.nullable() && !output_valid) out.set_null(i);
          return result;
        } else {
          return BinaryOperator{}.template operator()<TypeCommon, TypeCommon>(x, y);
        }
        // To supress nvcc warning
        return std::invoke_result_t<BinaryOperator, TypeCommon, TypeCommon>{};
      }();
      if constexpr (store_as_result)
        out.element<decltype(result)>(i) = result;
      else
        type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
    }
    (void)i;
  }
};

/**
 * @brief Functor to launch only defined operations without common type.
 *
 * @tparam BinaryOperator binary operator functor
 */
template <typename BinaryOperator, bool store_as_result = false>
struct ops2_wrapper {
  mutable_column_device_view& out;
  column_device_view const& lhs;
  column_device_view const& rhs;
  bool is_lhs_scalar;
  bool is_rhs_scalar;
  template <typename TypeLhs, typename TypeRhs>
  __device__ void operator()(size_type i)
  {
    if constexpr (!has_common_type_v<TypeLhs, TypeRhs> and
                  std::is_invocable_v<BinaryOperator, TypeLhs, TypeRhs>) {
      TypeLhs x   = lhs.element<TypeLhs>(is_lhs_scalar ? 0 : i);
      TypeRhs y   = rhs.element<TypeRhs>(is_rhs_scalar ? 0 : i);
      auto result = [&]() {
        if constexpr (std::is_same_v<BinaryOperator, ops::NullEquals> or
                      std::is_same_v<BinaryOperator, ops::NullMax> or
                      std::is_same_v<BinaryOperator, ops::NullMin>) {
          bool output_valid = false;
          auto result       = BinaryOperator{}.template operator()<TypeLhs, TypeRhs>(
            x,
            y,
            lhs.is_valid(is_lhs_scalar ? 0 : i),
            rhs.is_valid(is_rhs_scalar ? 0 : i),
            output_valid);
          if (out.nullable() && !output_valid) out.set_null(i);
          return result;
        } else {
          return BinaryOperator{}.template operator()<TypeLhs, TypeRhs>(x, y);
        }
        // To supress nvcc warning
        return std::invoke_result_t<BinaryOperator, TypeLhs, TypeRhs>{};
      }();
      if constexpr (store_as_result)
        out.element<decltype(result)>(i) = result;
      else
        type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
    }
    (void)i;
  }
};

/**
 * @brief Functor which does single, and double type dispatcher in device code
 *
 * single type dispatcher for lhs and rhs with common types.
 * double type dispatcher for lhs and rhs without common types.
 *
 * @tparam BinaryOperator binary operator functor
 */
template <class BinaryOperator, bool store_as_result = false>
struct device_type_dispatcher {
  mutable_column_device_view out;
  column_device_view lhs;
  column_device_view rhs;
  bool is_lhs_scalar;
  bool is_rhs_scalar;
  std::optional<data_type> common_data_type;

  __device__ void operator()(size_type i)
  {
    if (common_data_type) {
      type_dispatcher(
        *common_data_type,
        ops_wrapper<BinaryOperator, store_as_result>{out, lhs, rhs, is_lhs_scalar, is_rhs_scalar},
        i);
    } else {
      double_type_dispatcher(
        lhs.type(),
        rhs.type(),
        ops2_wrapper<BinaryOperator, store_as_result>{out, lhs, rhs, is_lhs_scalar, is_rhs_scalar},
        i);
    }
  }
};

template <class BinaryOperator>
void apply_binary_op(mutable_column_device_view& outd,
                     column_device_view const& lhsd,
                     column_device_view const& rhsd,
                     bool is_lhs_scalar,
                     bool is_rhs_scalar,
                     rmm::cuda_stream_view stream)
{
  auto common_dtype = get_common_type(outd.type(), lhsd.type(), rhsd.type());

  // Create binop functor instance
  auto binop_func = device_type_dispatcher<BinaryOperator>{
    outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, common_dtype};
  // Execute it on every element
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<size_type>(0),
                   thrust::make_counting_iterator<size_type>(outd.size()),
                   binop_func);
}

}  // namespace compiled
}  // namespace binops
}  // namespace cudf
