/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

/**
 * @brief Definition of the device operators
 * @file
 */

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/fixed_point/temporary.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/functional>

#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief SFINAE enabled min function suitable for std::is_invocable
 */
template <typename LHS,
          typename RHS,
          std::enable_if_t<cudf::is_relationally_comparable<LHS, RHS>()>* = nullptr>
CUDF_HOST_DEVICE inline auto min(LHS const& lhs, RHS const& rhs)
{
  return cuda::std::min(lhs, rhs);
}

/**
 * @brief SFINAE enabled max function suitable for std::is_invocable
 */
template <typename LHS,
          typename RHS,
          std::enable_if_t<cudf::is_relationally_comparable<LHS, RHS>()>* = nullptr>
CUDF_HOST_DEVICE inline auto max(LHS const& lhs, RHS const& rhs)
{
  return cuda::std::max(lhs, rhs);
}
}  // namespace detail

/**
 * @brief Binary `sum` operator
 */
struct DeviceSum {
  template <typename T, std::enable_if_t<!cudf::is_timestamp<T>()>* = nullptr>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs) -> decltype(lhs + rhs)
  {
    return lhs + rhs;
  }

  template <typename T, std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{typename T::duration{0}};
  }

  template <typename T,
            std::enable_if_t<!cudf::is_timestamp<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{0};
  }

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support device operator identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support device operator identity");
#endif
    return T{};
  }
};

/**
 * @brief `count` operator - used in rolling windows
 */
struct DeviceCount {
  template <typename T, std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
  CUDF_HOST_DEVICE inline T operator()(T const& lhs, T const& rhs)
  {
    return T{DeviceCount{}(lhs.time_since_epoch(), rhs.time_since_epoch())};
  }

  template <typename T, std::enable_if_t<!cudf::is_timestamp<T>()>* = nullptr>
  CUDF_HOST_DEVICE inline T operator()(T const&, T const& rhs)
  {
    return rhs + T{1};
  }

  template <typename T>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{};
  }
};

/**
 * @brief binary `min` operator
 */
struct DeviceMin {
  template <typename T>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs)
    -> decltype(cudf::detail::min(lhs, rhs))
  {
    return numeric::detail::min(lhs, rhs);
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_numeric<T>() && !cudf::is_fixed_point<T>())>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    if constexpr (cuda::std::numeric_limits<T>::has_infinity) {
      return cuda::std::numeric_limits<T>::infinity();
    } else {
      return cuda::std::numeric_limits<T>::max();
    }
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_point<T>())>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support DeviceMin identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support DeviceMin identity");
#endif
    return cuda::std::numeric_limits<T>::max();
  }

  // identity specialized for string_view and chrono types
  // chrono types do not have std::numeric_limits specializations and should use T::max()
  // https://eel.is/c++draft/numeric.limits.general#6
  template <typename T,
            CUDF_ENABLE_IF(cuda::std::is_same_v<T, cudf::string_view> || cudf::is_chrono<T>())>
  CUDF_HOST_DEVICE inline static constexpr T identity()
  {
    return T::max();
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_dictionary<T>())>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return static_cast<T>(T::max_value());
  }
};

/**
 * @brief binary `max` operator
 */
struct DeviceMax {
  template <typename T>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs)
    -> decltype(cudf::detail::max(lhs, rhs))
  {
    return numeric::detail::max(lhs, rhs);
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_numeric<T>() && !cudf::is_fixed_point<T>())>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    if constexpr (cuda::std::numeric_limits<T>::has_infinity) {
      return -cuda::std::numeric_limits<T>::infinity();
    } else {
      return cuda::std::numeric_limits<T>::lowest();
    }
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_point<T>())>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support DeviceMax identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support DeviceMax identity");
#endif
    return cuda::std::numeric_limits<T>::lowest();
  }

  // identity specialized for string_view and chrono types
  // chrono types do not have std::numeric_limits specializations and should use T::min()
  // https://eel.is/c++draft/numeric.limits.general#6
  template <typename T,
            CUDF_ENABLE_IF(cuda::std::is_same_v<T, cudf::string_view> || cudf::is_chrono<T>())>
  CUDF_HOST_DEVICE inline static constexpr T identity()
  {
    return T::min();
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_dictionary<T>())>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return static_cast<T>(T::lowest_value());
  }
};

/**
 * @brief binary `product` operator
 */
struct DeviceProduct {
  template <typename T, std::enable_if_t<!cudf::is_timestamp<T>()>* = nullptr>
  CUDF_HOST_DEVICE inline auto operator()(T const& lhs, T const& rhs) -> decltype(lhs * rhs)
  {
    return lhs * rhs;
  }

  template <typename T, std::enable_if_t<!cudf::is_fixed_point<T>()>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{1};
  }

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support DeviceProduct identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support DeviceProduct identity");
#endif
    return T{1, numeric::scale_type{0}};
  }
};

/**
 * @brief Operator for calculating Lead/Lag window function.
 */
struct DeviceLeadLag {
  size_type const row_offset;

  explicit CUDF_HOST_DEVICE inline DeviceLeadLag(size_type offset_) : row_offset(offset_) {}
};

/**
 * @brief Binary bitwise `AND` operator
 */
struct DeviceBitAnd {
  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  CUDF_HOST_DEVICE inline T operator()(T const& lhs, T const& rhs) const
  {
    return lhs & rhs;
  }

  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    if constexpr (std::is_same_v<T, bool>) {
      return true;
    } else {
      return ~T{0};
    }
  }

  template <typename T, std::enable_if_t<!std::is_integral_v<T>>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Bitwise AND is only supported for integral types.");
#else
    CUDF_UNREACHABLE("Bitwise AND is only supported for integral types.");
#endif
    return T{};
  }
};

/**
 * @brief Binary bitwise `OR` operator
 */
struct DeviceBitOr {
  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  CUDF_HOST_DEVICE inline T operator()(T const& lhs, T const& rhs) const
  {
    return lhs | rhs;
  }

  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{0};
  }

  template <typename T, std::enable_if_t<!std::is_integral_v<T>>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Bitwise OR is only supported for integral types.");
#else
    CUDF_UNREACHABLE("Bitwise OR is only supported for integral types.");
#endif
    return T{};
  }
};

/**
 * @brief Binary bitwise `XOR` operator
 */
struct DeviceBitXor {
  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  CUDF_HOST_DEVICE inline T operator()(T const& lhs, T const& rhs) const
  {
    return lhs ^ rhs;
  }

  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
    return T{0};
  }

  template <typename T, std::enable_if_t<!std::is_integral_v<T>>* = nullptr>
  CUDF_HOST_DEVICE static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Bitwise XOR is only supported for integral types.");
#else
    CUDF_UNREACHABLE("Bitwise XOR is only supported for integral types.");
#endif
    return T{};
  }
};

}  // namespace cudf
