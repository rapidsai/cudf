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

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>
#include "traits.hpp"

namespace cudf {
namespace binops {
namespace compiled {

template <typename CastType>
struct type_casted_accessor {
  template <typename Element>
  CUDA_DEVICE_CALLABLE CastType operator()(cudf::size_type i, column_device_view const col) const
  {
    if constexpr (column_device_view::has_element_accessor<Element>() and
                  std::is_convertible_v<Element, CastType>)
      return static_cast<CastType>(col.element<Element>(i));
    return {};
  }
};

template <typename FromType>
struct typed_casted_writer {
  template <typename Element>
  CUDA_DEVICE_CALLABLE void operator()(cudf::size_type i,
                                       mutable_column_device_view const col,
                                       FromType val) const
  {
    if constexpr (mutable_column_device_view::has_element_accessor<Element>() and
                  std::is_constructible_v<Element, FromType>) {
      col.element<Element>(i) = static_cast<Element>(val);
    }
  }
};

// TODO use inheritance. (or some other way to simplify the operator.)
struct Add {
  template <typename TypeCommon>
  static constexpr inline bool is_supported()
  {
    return cudf::binops::compiled::CHECK::PlusExists<TypeCommon, TypeCommon>::value;
  }

  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return !has_common_type_v<TypeLhs, TypeRhs> and
           cudf::binops::compiled::CHECK::PlusExists<TypeLhs, TypeRhs>::value;
    //(is_chrono<TypeLhs>() and is_chrono<TypeRhs>()) and
    //!(is_timestamp<TypeLhs>() and is_timestamp<TypeRhs>());
  }

  // 1. With common type. (single dispatch, + typecast dispatch)
  // Allow sum between chronos only when both input and output types
  // are chronos. Unsupported combinations will fail to compile
  template <typename TypeCommon, std::enable_if_t<is_supported<TypeCommon>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeCommon x = type_dispatcher(lhs.type(), type_casted_accessor<TypeCommon>{}, i, lhs);
    TypeCommon y = type_dispatcher(rhs.type(), type_casted_accessor<TypeCommon>{}, i, rhs);
    auto result  = x + y;
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }

  // 2. without common type. (double dispatch, + out typecast dispatch)

  // chronos t+d, d+t, d+d, !(t+t)
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<is_supported<TypeLhs, TypeRhs>()
                             //(is_chrono<TypeLhs>() and is_chrono<TypeRhs>()) and
                             //!(is_timestamp<TypeLhs>() and is_timestamp<TypeRhs>())
                             >* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeLhs x   = lhs.element<TypeLhs>(i);
    TypeRhs y   = rhs.element<TypeRhs>(i);
    auto result = x + y;
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }
};

struct Sub {
  template <typename TypeCommon>
  static constexpr inline bool is_supported()
  {
    return cudf::binops::compiled::CHECK::MinusExists<TypeCommon, TypeCommon>::value;
  }

  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return (!has_common_type_v<TypeLhs, TypeRhs>)and cudf::binops::compiled::CHECK::
      MinusExists<TypeLhs, TypeRhs>::value;
  }

  // Allow sum between chronos only when both input and output types
  // are chronos. Unsupported combinations will fail to compile
  template <typename TypeCommon, std::enable_if_t<is_supported<TypeCommon>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeCommon x = type_dispatcher(lhs.type(), type_casted_accessor<TypeCommon>{}, i, lhs);
    TypeCommon y = type_dispatcher(rhs.type(), type_casted_accessor<TypeCommon>{}, i, rhs);
    auto result  = x - y;
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }
  // chronos t-d, d-d, t-t, !(d-t)
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<is_supported<TypeLhs, TypeRhs>()
                             //(is_chrono<TypeLhs>() and is_chrono<TypeRhs>()) and
                             //!(is_duration<TypeLhs>() and is_timestamp<TypeRhs>())
                             >* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeLhs x   = lhs.element<TypeLhs>(i);
    TypeRhs y   = rhs.element<TypeRhs>(i);
    auto result = x - y;
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }
};

struct Mul {
  template <typename TypeCommon>
  static constexpr inline bool is_supported()
  {
    return cudf::binops::compiled::CHECK::MulExists<TypeCommon, TypeCommon>::value;
  }

  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return (!has_common_type_v<TypeLhs, TypeRhs>)and cudf::binops::compiled::CHECK::
             MulExists<TypeLhs, TypeRhs>::value and
           ((is_duration<TypeLhs>() && std::is_integral<TypeRhs>()) ||
            (std::is_integral<TypeLhs>() && is_duration<TypeRhs>()));
  }

  // Allow sum between chronos only when both input and output types
  // are chronos. Unsupported combinations will fail to compile
  template <typename TypeCommon, std::enable_if_t<is_supported<TypeCommon>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeCommon x = type_dispatcher(lhs.type(), type_casted_accessor<TypeCommon>{}, i, lhs);
    TypeCommon y = type_dispatcher(rhs.type(), type_casted_accessor<TypeCommon>{}, i, rhs);
    auto result  = x * y;
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }
  // chronos n*d, d*n
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<is_supported<TypeLhs, TypeRhs>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeLhs x   = lhs.element<TypeLhs>(i);
    TypeRhs y   = rhs.element<TypeRhs>(i);
    auto result = x * y;
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }
};

struct Div {
  template <typename TypeCommon>
  static constexpr inline bool is_supported()
  {
    return cudf::binops::compiled::CHECK::DivExists<TypeCommon, TypeCommon>::value;
  }

  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return (!has_common_type_v<TypeLhs, TypeRhs>)and cudf::binops::compiled::CHECK::
             DivExists<TypeLhs, TypeRhs>::value and
           is_duration<TypeLhs>() && (std::is_integral<TypeRhs>() || is_duration<TypeRhs>());
  }

  // Allow sum between chronos only when both input and output types
  // are chronos. Unsupported combinations will fail to compile
  template <typename TypeCommon, std::enable_if_t<is_supported<TypeCommon>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeCommon x = type_dispatcher(lhs.type(), type_casted_accessor<TypeCommon>{}, i, lhs);
    TypeCommon y = type_dispatcher(rhs.type(), type_casted_accessor<TypeCommon>{}, i, rhs);
    auto result  = x / y;
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }
  // chronos d/n, d/d
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<is_supported<TypeLhs, TypeRhs>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeLhs x   = lhs.element<TypeLhs>(i);
    TypeRhs y   = rhs.element<TypeRhs>(i);
    auto result = x / y;
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }
};

struct TrueDiv {
  template <typename TypeCommon>
  static constexpr inline bool is_supported()
  {
    return std::is_constructible_v<double, TypeCommon>;
  }

  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return (!has_common_type_v<TypeLhs, TypeRhs>)and std::is_constructible_v<double, TypeLhs> and
           std::is_constructible_v<double, TypeRhs>;
  }

  // Allow sum between chronos only when both input and output types
  // are chronos. Unsupported combinations will fail to compile
  template <typename TypeCommon, std::enable_if_t<is_supported<TypeCommon>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeCommon x = type_dispatcher(lhs.type(), type_casted_accessor<TypeCommon>{}, i, lhs);
    TypeCommon y = type_dispatcher(rhs.type(), type_casted_accessor<TypeCommon>{}, i, rhs);
    auto result  = (static_cast<double>(x) / static_cast<double>(y));
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }
  // chronos d/n, d/d
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<is_supported<TypeLhs, TypeRhs>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(size_type i,
                                            column_device_view const lhs,
                                            column_device_view const rhs,
                                            mutable_column_device_view const out)
  {
    TypeLhs x   = lhs.element<TypeLhs>(i);
    TypeRhs y   = rhs.element<TypeRhs>(i);
    auto result = (static_cast<double>(x) / static_cast<double>(y));
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }
};

struct Mul2 {
  // Allow sum between chronos only when both input and output types
  // are chronos. Unsupported combinations will fail to compile
  template <typename TypeOut,
            typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<
              (CHECK::MulExists<TypeLhs, TypeRhs>::value and
               std::is_convertible<decltype(std::declval<TypeLhs>() * std::declval<TypeRhs>()),
                                   TypeOut>::value)>* = nullptr>
  static CUDA_HOST_DEVICE_CALLABLE TypeOut operate(TypeLhs x, TypeRhs y)
  {
    if constexpr (has_common_type_v<TypeOut, TypeLhs, TypeRhs>) {
      using TypeCommon = typename std::common_type<TypeOut, TypeLhs, TypeRhs>::type;
      return static_cast<TypeOut>(static_cast<TypeCommon>(x) * static_cast<TypeCommon>(y));
    } else {
      return static_cast<TypeOut>(x * y);
    }
  }
};

data_type get_common_type(data_type out, data_type lhs, data_type rhs);

}  // namespace compiled
}  // namespace binops
}  // namespace cudf
