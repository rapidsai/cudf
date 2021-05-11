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

data_type get_common_type(data_type out, data_type lhs, data_type rhs);

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

namespace ops {

struct Add {
  template <typename T1, typename T2>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs + rhs)
  {
    return lhs + rhs;
  }
};

struct Sub {
  template <typename T1, typename T2>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs - rhs)
  {
    return lhs - rhs;
  }
};

struct Mul {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without the following line, compilation error
           // _deps/libcudacxx-src/include/cuda/std/detail/libcxx/include/chrono(917): error:
           // identifier "cuda::std::__3::ratio<(long)86400000000l, (long)1l> ::num" is undefined in
           // device code
           ((is_duration<TypeLhs>() and std::is_integral<TypeRhs>()) or
            (std::is_integral<TypeLhs>() and is_duration<TypeRhs>()));
  }
  template <typename T1, typename T2, std::enable_if_t<is_supported<T1, T2>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs * rhs)
  {
    return lhs * rhs;
  }
};

struct Div {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without this, compilation error on chrono:917
           is_duration<TypeLhs>() and (std::is_integral<TypeRhs>() or is_duration<TypeRhs>());
  }
  template <typename T1, typename T2, std::enable_if_t<is_supported<T1, T2>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs / rhs)
  {
    return lhs / rhs;
  }
};

struct TrueDiv {
  template <typename T1, typename T2>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs)
    -> decltype((static_cast<double>(lhs) / static_cast<double>(rhs)))
  {
    return (static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

struct FloorDiv {
  template <typename T1, typename T2>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs)
    -> decltype(floor(static_cast<double>(lhs) / static_cast<double>(rhs)))
  {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

struct Mod {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without this, compilation error
           //_deps/libcudacxx-src/include/cuda/std/detail/libcxx/include/chrono(1337):
           // error : expression must have integral or unscoped enum type
           is_duration<TypeLhs>() and (std::is_integral<TypeRhs>() or is_duration<TypeRhs>());
  }
  template <typename T1, typename T2, std::enable_if_t<is_supported<T1, T2>()>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs % rhs)
  {
    return lhs % rhs;
  }
};
}  // namespace ops
}  // namespace compiled
}  // namespace binops
}  // namespace cudf
