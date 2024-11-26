/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace binops {
namespace compiled {

template <typename BinaryOperator, typename TypeLhs, typename TypeRhs>
constexpr bool is_bool_result()
{
  using ReturnType = std::invoke_result_t<BinaryOperator, TypeLhs, TypeRhs>;
  return std::is_same_v<bool, ReturnType>;
}

/**
 * @brief Type casts each element of the column to `CastType`
 *
 */
template <typename CastType>
struct type_casted_accessor {
  template <typename Element>
  __device__ inline CastType operator()(cudf::size_type i,
                                        column_device_view const& col,
                                        bool is_scalar) const
  {
    if constexpr (column_device_view::has_element_accessor<Element>()) {
      auto const element = col.element<Element>(is_scalar ? 0 : i);
      if constexpr (std::is_convertible_v<Element, CastType>) {
        return static_cast<CastType>(element);
      } else if constexpr (is_fixed_point<Element>() && cuda::std::is_floating_point_v<CastType>) {
        return convert_fixed_to_floating<CastType>(element);
      } else if constexpr (is_fixed_point<CastType>() && cuda::std::is_floating_point_v<Element>) {
        return convert_floating_to_fixed<CastType>(element, numeric::scale_type{0});
      }
    }
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
  __device__ inline void operator()(cudf::size_type i,
                                    mutable_column_device_view const& col,
                                    FromType val) const
  {
    if constexpr (mutable_column_device_view::has_element_accessor<Element>() and
                  std::is_constructible_v<Element, FromType>) {
      col.element<Element>(i) = static_cast<Element>(val);
    } else if constexpr (is_fixed_point<Element>()) {
      auto const scale = numeric::scale_type{col.type().scale()};
      if constexpr (is_fixed_point<FromType>()) {
        col.data<Element::rep>()[i] = val.rescaled(scale).value();
      } else if constexpr (cuda::std::is_constructible_v<Element, FromType>) {
        col.data<Element::rep>()[i] = Element{val, scale}.value();
      } else if constexpr (cuda::std::is_floating_point_v<FromType>) {
        col.data<Element::rep>()[i] = convert_floating_to_fixed<Element>(val, scale).value();
      }
    } else if constexpr (cuda::std::is_floating_point_v<Element> and is_fixed_point<FromType>()) {
      col.data<Element>()[i] = convert_fixed_to_floating<Element>(val);
    }
  }
};

// Functors to launch only defined operations.

/**
 * @brief Functor to launch only defined operations with common type.
 *
 * @tparam BinaryOperator binary operator functor
 */
template <typename BinaryOperator>
struct ops_wrapper {
  mutable_column_device_view& out;
  column_device_view const& lhs;
  column_device_view const& rhs;
  bool const& is_lhs_scalar;
  bool const& is_rhs_scalar;
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
                      std::is_same_v<BinaryOperator, ops::NullNotEquals> or
                      std::is_same_v<BinaryOperator, ops::NullLogicalAnd> or
                      std::is_same_v<BinaryOperator, ops::NullLogicalOr> or
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
        // To suppress nvcc warning
        return std::invoke_result_t<BinaryOperator, TypeCommon, TypeCommon>{};
      }();
      if constexpr (is_bool_result<BinaryOperator, TypeCommon, TypeCommon>())
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
template <typename BinaryOperator>
struct ops2_wrapper {
  mutable_column_device_view& out;
  column_device_view const& lhs;
  column_device_view const& rhs;
  bool const& is_lhs_scalar;
  bool const& is_rhs_scalar;
  template <typename TypeLhs, typename TypeRhs>
  __device__ void operator()(size_type i)
  {
    if constexpr (!has_common_type_v<TypeLhs, TypeRhs> and
                  std::is_invocable_v<BinaryOperator, TypeLhs, TypeRhs>) {
      TypeLhs x   = lhs.element<TypeLhs>(is_lhs_scalar ? 0 : i);
      TypeRhs y   = rhs.element<TypeRhs>(is_rhs_scalar ? 0 : i);
      auto result = [&]() {
        if constexpr (std::is_same_v<BinaryOperator, ops::NullEquals> or
                      std::is_same_v<BinaryOperator, ops::NullNotEquals> or
                      std::is_same_v<BinaryOperator, ops::NullLogicalAnd> or
                      std::is_same_v<BinaryOperator, ops::NullLogicalOr> or
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
        // To suppress nvcc warning
        return std::invoke_result_t<BinaryOperator, TypeLhs, TypeRhs>{};
      }();
      if constexpr (is_bool_result<BinaryOperator, TypeLhs, TypeRhs>())
        out.element<decltype(result)>(i) = result;
      else
        type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
    }
    (void)i;
  }
};

/**
 * @brief Functor which does single type dispatcher in device code
 *
 * single type dispatcher for lhs and rhs with common types.
 *
 * @tparam BinaryOperator binary operator functor
 */
template <class BinaryOperator>
struct binary_op_device_dispatcher {
  data_type common_data_type;
  mutable_column_device_view out;
  column_device_view lhs;
  column_device_view rhs;
  bool is_lhs_scalar;
  bool is_rhs_scalar;

  __forceinline__ __device__ void operator()(size_type i)
  {
    type_dispatcher(common_data_type,
                    ops_wrapper<BinaryOperator>{out, lhs, rhs, is_lhs_scalar, is_rhs_scalar},
                    i);
  }
};

/**
 * @brief Functor which does double type dispatcher in device code
 *
 * double type dispatcher for lhs and rhs without common types.
 *
 * @tparam BinaryOperator binary operator functor
 */
template <class BinaryOperator>
struct binary_op_double_device_dispatcher {
  mutable_column_device_view out;
  column_device_view lhs;
  column_device_view rhs;
  bool is_lhs_scalar;
  bool is_rhs_scalar;

  __forceinline__ __device__ void operator()(size_type i)
  {
    double_type_dispatcher(
      lhs.type(),
      rhs.type(),
      ops2_wrapper<BinaryOperator>{out, lhs, rhs, is_lhs_scalar, is_rhs_scalar},
      i);
  }
};

template <class BinaryOperator>
void apply_binary_op(mutable_column_view& out,
                     column_view const& lhs,
                     column_view const& rhs,
                     bool is_lhs_scalar,
                     bool is_rhs_scalar,
                     rmm::cuda_stream_view stream)
{
  auto common_dtype = get_common_type(out.type(), lhs.type(), rhs.type());

  auto lhsd = column_device_view::create(lhs, stream);
  auto rhsd = column_device_view::create(rhs, stream);
  auto outd = mutable_column_device_view::create(out, stream);
  // Create binop functor instance
  if (common_dtype) {
    // Execute it on every element
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::counting_iterator<size_type>(0),
                       out.size(),
                       binary_op_device_dispatcher<BinaryOperator>{
                         *common_dtype, *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
  } else {
    // Execute it on every element
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::counting_iterator<size_type>(0),
                       out.size(),
                       binary_op_double_device_dispatcher<BinaryOperator>{
                         *outd, *lhsd, *rhsd, is_lhs_scalar, is_rhs_scalar});
  }
}

}  // namespace compiled
}  // namespace binops
}  // namespace cudf
