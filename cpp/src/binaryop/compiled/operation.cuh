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

#include <cudf/utilities/traits.hpp>

#include <cmath>

namespace cudf {
namespace binops {
namespace compiled {

// All binary operations
namespace ops {

struct Add {
  template <typename T1, typename T2>
  CUDA_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs + rhs)
  {
    return lhs + rhs;
  }
};

struct Sub {
  template <typename T1, typename T2>
  CUDA_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs - rhs)
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
           (is_duration<TypeLhs>() and std::is_integral<TypeRhs>()) or
           (std::is_integral<TypeLhs>() and is_duration<TypeRhs>()) or
           (is_fixed_point<TypeLhs>() and is_numeric<TypeRhs>()) or
           (is_numeric<TypeLhs>() and is_fixed_point<TypeRhs>());
  }
  template <typename T1, typename T2, std::enable_if_t<is_supported<T1, T2>()>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs * rhs)
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
           (is_duration<TypeLhs>() and (std::is_integral<TypeRhs>() or is_duration<TypeRhs>())) or
           (is_fixed_point<TypeLhs>() and is_numeric<TypeRhs>()) or
           (is_numeric<TypeLhs>() and is_fixed_point<TypeRhs>());
  }
  template <typename T1, typename T2, std::enable_if_t<is_supported<T1, T2>()>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs / rhs)
  {
    return lhs / rhs;
  }
};

struct TrueDiv {
  template <typename T1, typename T2>
  CUDA_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs)
    -> decltype((static_cast<double>(lhs) / static_cast<double>(rhs)))
  {
    return (static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

struct FloorDiv {
  template <typename T1, typename T2>
  CUDA_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs)
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
           (is_duration<TypeLhs>() and (std::is_integral<TypeRhs>() or is_duration<TypeRhs>()));
  }
  template <typename T1, typename T2, std::enable_if_t<is_supported<T1, T2>()>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs % rhs)
  {
    return lhs % rhs;
  }
  template <typename T1,
            typename T2,
            std::enable_if_t<(std::is_same_v<float, std::common_type_t<T1, T2>>)>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> float
  {
    return fmodf(static_cast<float>(lhs), static_cast<float>(rhs));
  }
  template <typename T1,
            typename T2,
            std::enable_if_t<(std::is_same_v<double, std::common_type_t<T1, T2>>)>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(T1 const& lhs, T2 const& rhs) -> double
  {
    return fmod(static_cast<double>(lhs), static_cast<double>(rhs));
  }
};

struct PMod {
  // Ideally, these two specializations - one for integral types and one for non integral
  // types shouldn't be required, as std::fmod should promote integral types automatically
  // to double and call the std::fmod overload for doubles. Sadly, doing this in jitified
  // code does not work - it is having trouble deciding between float/double overloads
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<(std::is_integral_v<std::common_type_t<TypeLhs, TypeRhs>>)>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y)
  {
    using common_t = std::common_type_t<TypeLhs, TypeRhs>;
    common_t xconv = static_cast<common_t>(x);
    common_t yconv = static_cast<common_t>(y);
    auto rem       = xconv % yconv;
    if constexpr (std::is_signed_v<decltype(rem)>)
      if (rem < 0) rem = (rem + yconv) % yconv;
    return rem;
  }

  template <
    typename TypeLhs,
    typename TypeRhs,
    std::enable_if_t<(std::is_floating_point_v<std::common_type_t<TypeLhs, TypeRhs>>)>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y)
  {
    using common_t = std::common_type_t<TypeLhs, TypeRhs>;
    common_t xconv = static_cast<common_t>(x);
    common_t yconv = static_cast<common_t>(y);
    auto rem       = std::fmod(xconv, yconv);
    if (rem < 0) rem = std::fmod(rem + yconv, yconv);
    return rem;
  }
};

struct PyMod {
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<(std::is_integral_v<std::common_type_t<TypeLhs, TypeRhs>>)>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(((x % y) + y) % y)
  {
    return ((x % y) + y) % y;
  }

  template <
    typename TypeLhs,
    typename TypeRhs,
    std::enable_if_t<(std::is_floating_point_v<std::common_type_t<TypeLhs, TypeRhs>>)>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> double
  {
    double x1 = static_cast<double>(x);
    double y1 = static_cast<double>(y);
    return fmod(fmod(x1, y1) + y1, y1);
  }

  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<(is_duration<TypeLhs>())>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(((x % y) + y) % y)
  {
    return ((x % y) + y) % y;
  }
};

struct Pow {
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<(std::is_convertible_v<TypeLhs, double> and
                              std::is_convertible_v<TypeRhs, double>)>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> double
  {
    return pow(static_cast<double>(x), static_cast<double>(y));
  }
};

struct LogBase {
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<(std::is_convertible_v<TypeLhs, double> and
                              std::is_convertible_v<TypeRhs, double>)>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> double
  {
    return (std::log(static_cast<double>(x)) / std::log(static_cast<double>(y)));
  }
};

struct ATan2 {
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<(std::is_convertible_v<TypeLhs, double> and
                              std::is_convertible_v<TypeRhs, double>)>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> double
  {
    return std::atan2(static_cast<double>(x), static_cast<double>(y));
  }
};

struct ShiftLeft {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x << y)
  {
    return (x << y);
  }
};

struct ShiftRight {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x >> y)
  {
    return (x >> y);
  }
};

struct ShiftRightUnsigned {
  template <
    typename TypeLhs,
    typename TypeRhs,
    std::enable_if_t<(std::is_integral_v<TypeLhs> and not is_boolean<TypeLhs>())>* = nullptr>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y)
    -> decltype(static_cast<std::make_unsigned_t<TypeLhs>>(x) >> y)
  {
    return (static_cast<std::make_unsigned_t<TypeLhs>>(x) >> y);
  }
};

struct BitwiseAnd {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x & y)
  {
    return (x & y);
  }
};

struct BitwiseOr {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x | y)
  {
    return (x | y);
  }
};

struct BitwiseXor {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x ^ y)
  {
    return (x ^ y);
  }
};

struct LogicalAnd {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x && y)
  {
    return (x && y);
  }
};

struct LogicalOr {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x || y)
  {
    return (x || y);
  }
};

struct Equal {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x == y)
  {
    return (x == y);
  }
};

struct NotEqual {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x != y)
  {
    return (x != y);
  }
};

struct Less {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x < y)
  {
    return (x < y);
  }
};

struct Greater {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x > y)
  {
    return (x > y);
  }
};

struct LessEqual {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x <= y)
  {
    return (x <= y);
  }
};

struct GreaterEqual {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x >= y)
  {
    return (x >= y);
  }
};

struct NullEquals {
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) -> decltype(x == y)
  {
    output_valid = true;
    if (!lhs_valid && !rhs_valid) return true;
    if (lhs_valid && rhs_valid) return x == y;
    return false;
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs, typename TypeRhs>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y) -> decltype(x == y);
};

struct NullMax {
  template <typename TypeLhs,
            typename TypeRhs,
            typename common_t = std::common_type_t<TypeLhs, TypeRhs>>
  CUDA_DEVICE_CALLABLE auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid)
    -> decltype(static_cast<common_t>(static_cast<common_t>(x) > static_cast<common_t>(y) ? x : y))
  {
    output_valid      = true;
    auto const x_conv = static_cast<common_t>(x);
    auto const y_conv = static_cast<common_t>(y);
    if (!lhs_valid && !rhs_valid) {
      output_valid = false;
      return common_t{};
    } else if (lhs_valid && rhs_valid) {
      return (x_conv > y_conv) ? x_conv : y_conv;
    } else if (lhs_valid)
      return x_conv;
    else
      return y_conv;
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs,
            typename TypeRhs,
            typename common_t = std::common_type_t<TypeLhs, TypeRhs>>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y)
    -> decltype(static_cast<common_t>(static_cast<common_t>(x) > static_cast<common_t>(y) ? x : y));
};

struct NullMin {
  template <typename TypeLhs,
            typename TypeRhs,
            typename common_t = std::common_type_t<TypeLhs, TypeRhs>>
  CUDA_DEVICE_CALLABLE auto operator()(
    TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid)
    -> decltype(static_cast<common_t>(static_cast<common_t>(x) < static_cast<common_t>(y) ? x : y))
  {
    output_valid      = true;
    auto const x_conv = static_cast<common_t>(x);
    auto const y_conv = static_cast<common_t>(y);
    if (!lhs_valid && !rhs_valid) {
      output_valid = false;
      return common_t{};
    } else if (lhs_valid && rhs_valid) {
      return (x_conv < y_conv) ? x_conv : y_conv;
    } else if (lhs_valid)
      return x_conv;
    else
      return y_conv;
  }
  // To allow std::is_invocable_v = true
  template <typename TypeLhs,
            typename TypeRhs,
            typename common_t = std::common_type_t<TypeLhs, TypeRhs>>
  CUDA_DEVICE_CALLABLE auto operator()(TypeLhs x, TypeRhs y)
    -> decltype(static_cast<common_t>(static_cast<common_t>(x) < static_cast<common_t>(y) ? x : y));
};

}  // namespace ops
}  // namespace compiled
}  // namespace binops
}  // namespace cudf
