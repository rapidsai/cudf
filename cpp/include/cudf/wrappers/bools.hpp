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

#include <utilities/cudf_utils.h>

#include <cub/util_type.cuh>

#include <cstdint>
#include <iosfwd>

/**---------------------------------------------------------------------------*
 * @file bools.hpp
 * @brief Concrete type definition for uint8_t values !0 and 0 representing
 * boolean true and false respectively.
 *---------------------------------------------------------------------------**/
namespace cudf {
namespace experimental {

struct bool8 {

  uint8_t value;

  template <typename from_type>
  CUDA_HOST_DEVICE_CALLABLE constexpr explicit bool8(from_type v)
    : value{static_cast<bool>(v)} {}

  bool8(bool8 const& w) = default;
  bool8() = default;

  // cudf::experimental::bool8 bool conversion operators

  CUDA_HOST_DEVICE_CALLABLE explicit operator uint8_t() const {
    return this->value;
  }

  CUDA_HOST_DEVICE_CALLABLE explicit operator bool() const {
    return static_cast<bool>(this->value);
  }

  // cudf::experimental::bool8 ostream << operator overload

  inline std::ostream& operator<<(std::ostream& os) {
    return os << static_cast<bool>(this->value);
  }

  // cudf::experimental::bool8 binary operator overloads

  CUDA_HOST_DEVICE_CALLABLE bool operator==(cudf::experimental::bool8 const &rhs) const {
    return static_cast<bool>(*this) == static_cast<bool>(rhs);
  }

  CUDA_HOST_DEVICE_CALLABLE bool operator!=(cudf::experimental::bool8 const &rhs) const {
    return static_cast<bool>(*this) != static_cast<bool>(rhs);
  }

  CUDA_HOST_DEVICE_CALLABLE bool operator<=(cudf::experimental::bool8 const &rhs) const {
    return static_cast<bool>(*this) <= static_cast<bool>(rhs); 
  }

  CUDA_HOST_DEVICE_CALLABLE bool operator>=(cudf::experimental::bool8 const &rhs) const {
    return static_cast<bool>(*this) >= static_cast<bool>(rhs); 
  }

  CUDA_HOST_DEVICE_CALLABLE bool operator<(cudf::experimental::bool8 const &rhs) const {
    return static_cast<bool>(*this) < static_cast<bool>(rhs);
  }

  CUDA_HOST_DEVICE_CALLABLE bool operator>(cudf::experimental::bool8 const &rhs) const {
    return static_cast<bool>(*this) > static_cast<bool>(rhs);
  }

  CUDA_HOST_DEVICE_CALLABLE cudf::experimental::bool8 operator+(cudf::experimental::bool8 const &rhs) const {
    return static_cast<cudf::experimental::bool8>(static_cast<bool>(*this) +
                                         static_cast<bool>(rhs));
  }

  CUDA_HOST_DEVICE_CALLABLE cudf::experimental::bool8 operator-(cudf::experimental::bool8 const &rhs) const {
    return static_cast<cudf::experimental::bool8>(static_cast<bool>(*this) -
                                         static_cast<bool>(rhs));
  }

  CUDA_HOST_DEVICE_CALLABLE cudf::experimental::bool8 operator*(cudf::experimental::bool8 const &rhs) const {
    return static_cast<cudf::experimental::bool8>(static_cast<bool>(*this) *
                                         static_cast<bool>(rhs));
  }

  CUDA_HOST_DEVICE_CALLABLE cudf::experimental::bool8 operator/(cudf::experimental::bool8 const &rhs) const {
    return static_cast<cudf::experimental::bool8>(static_cast<bool>(*this) /
                                         static_cast<bool>(rhs));
  }

  // cudf::experimental::bool8 unary operator overloads

  CUDA_HOST_DEVICE_CALLABLE cudf::experimental::bool8& operator+=(cudf::experimental::bool8 const &rhs) {
    cudf::experimental::bool8 &lhs = *this;
    lhs = lhs + rhs;
    return lhs;
  }

  CUDA_HOST_DEVICE_CALLABLE cudf::experimental::bool8& operator-=(cudf::experimental::bool8 const &rhs) {
    cudf::experimental::bool8 &lhs = *this;
    lhs = lhs - rhs;
    return lhs;
  }

  CUDA_HOST_DEVICE_CALLABLE cudf::experimental::bool8& operator*=(cudf::experimental::bool8 const &rhs) {
    cudf::experimental::bool8 &lhs = *this;
    lhs = lhs * rhs;
    return lhs;
  }

  CUDA_HOST_DEVICE_CALLABLE cudf::experimental::bool8& operator/=(cudf::experimental::bool8 const &rhs) {
    cudf::experimental::bool8 &lhs = *this;
    lhs = lhs / rhs;
    return lhs;
  }

  CUDA_HOST_DEVICE_CALLABLE cudf::experimental::bool8 operator!() const {
    return static_cast<cudf::experimental::bool8>(!static_cast<bool>(*this));
  }
};

// This is necessary for global, constant, non-fundamental types
// We can't rely on --expt-relaxed-constexpr here because `bool8` is not a
// scalar type. See CUDA Programming guide
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-variables
#ifdef __CUDA_ARCH__
__device__ __constant__ static bool8 true_v{uint8_t{1}};
__device__ __constant__ static bool8 false_v{uint8_t{0}};
#else
static constexpr bool8 true_v{uint8_t{1}};
static constexpr bool8 false_v{uint8_t{0}};
#endif

} // experimental
}  // cudf

namespace std {
/** --------------------------------------------------------------------------*
  * @brief Specialization of std::numeric_limits for cudf::experimental::bool8
  *
  * Required since the underlying type, uint8_t, has different limits than bool
  * --------------------------------------------------------------------------**/
template <>
struct numeric_limits<cudf::experimental::bool8> {
  
  static constexpr cudf::experimental::bool8 max() noexcept {
    // tried using `return cudf::true_v` but it causes a compiler segfault!
    return cudf::experimental::bool8{true};
  }
  
  static constexpr cudf::experimental::bool8 lowest() noexcept {
    return cudf::experimental::bool8{false};
  }

  static constexpr cudf::experimental::bool8 min() noexcept {
    return cudf::experimental::bool8{false};
  }
};

} // std

namespace cub {

template <> struct NumericTraits<cudf::experimental::bool8> :
  BaseTraits<SIGNED_INTEGER, true, false, uint8_t, int8_t> {};

} // cub
