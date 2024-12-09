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

/**
 * @brief Definition of the device operators
 * @file
 */

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/fixed_point/temporary.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

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
  return std::min(lhs, rhs);
}

/**
 * @brief SFINAE enabled max function suitable for std::is_invocable
 */
template <typename LHS,
          typename RHS,
          std::enable_if_t<cudf::is_relationally_comparable<LHS, RHS>()>* = nullptr>
CUDF_HOST_DEVICE inline auto max(LHS const& lhs, RHS const& rhs)
{
  return std::max(lhs, rhs);
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
  static constexpr T identity()
  {
    return T{typename T::duration{0}};
  }

  template <typename T,
            std::enable_if_t<!cudf::is_timestamp<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  static constexpr T identity()
  {
    return T{0};
  }

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  static constexpr T identity()
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
  static constexpr T identity()
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

  template <typename T,
            std::enable_if_t<!std::is_same_v<T, cudf::string_view> && !cudf::is_dictionary<T>() &&
                             !cudf::is_fixed_point<T>()>* = nullptr>
  static constexpr T identity()
  {
    // chrono types do not have std::numeric_limits specializations and should use T::max()
    // https://eel.is/c++draft/numeric.limits.general#6
    if constexpr (cudf::is_chrono<T>()) {
      return T::max();
    } else if constexpr (cuda::std::numeric_limits<T>::has_infinity) {
      return cuda::std::numeric_limits<T>::infinity();
    } else {
      return cuda::std::numeric_limits<T>::max();
    }
  }

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support DeviceMin identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support DeviceMin identity");
#endif
    return cuda::std::numeric_limits<T>::max();
  }

  // @brief identity specialized for string_view
  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>* = nullptr>
  CUDF_HOST_DEVICE inline static constexpr T identity()
  {
    return string_view::max();
  }

  template <typename T, std::enable_if_t<cudf::is_dictionary<T>()>* = nullptr>
  static constexpr T identity()
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

  template <typename T,
            std::enable_if_t<!std::is_same_v<T, cudf::string_view> && !cudf::is_dictionary<T>() &&
                             !cudf::is_fixed_point<T>()>* = nullptr>
  static constexpr T identity()
  {
    // chrono types do not have std::numeric_limits specializations and should use T::min()
    // https://eel.is/c++draft/numeric.limits.general#6
    if constexpr (cudf::is_chrono<T>()) {
      return T::min();
    } else if constexpr (cuda::std::numeric_limits<T>::has_infinity) {
      return -cuda::std::numeric_limits<T>::infinity();
    } else {
      return cuda::std::numeric_limits<T>::lowest();
    }
  }

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  static constexpr T identity()
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("fixed_point does not yet support DeviceMax identity");
#else
    CUDF_UNREACHABLE("fixed_point does not yet support DeviceMax identity");
#endif
    return cuda::std::numeric_limits<T>::lowest();
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>* = nullptr>
  CUDF_HOST_DEVICE inline static constexpr T identity()
  {
    return string_view::min();
  }

  template <typename T, std::enable_if_t<cudf::is_dictionary<T>()>* = nullptr>
  static constexpr T identity()
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
  static constexpr T identity()
  {
    return T{1};
  }

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  static constexpr T identity()
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

}  // namespace cudf
