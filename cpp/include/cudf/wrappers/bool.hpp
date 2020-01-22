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

#include <cudf/types.hpp>

#include <cuda_runtime.h>
#include <cub/util_type.cuh>

#include <cstdint>
#include <iosfwd>
#include <type_traits>

/**---------------------------------------------------------------------------*
 * @file bools.hpp
 * @brief Concrete type definition for uint8_t values !0 and 0 representing
 * boolean true and false respectively.
 *---------------------------------------------------------------------------**/
namespace cudf {
namespace experimental {

struct bool8 {

  using value_type = uint8_t;

  // defaulted bool8 move/copy constructors

  bool8() = default;
  ~bool8() = default;
  bool8(bool8 &&) = default;
  bool8(bool8 const& w) = default;
  bool8& operator=(bool8&&) = default;
  bool8& operator=(const bool8&) = default;

  CUDA_HOST_DEVICE_CALLABLE constexpr bool8(bool v)
    : value{static_cast<value_type>(v)} {}

  // move/copy assignment operators for non-bool8 types

  template <typename from_type>
  CUDA_HOST_DEVICE_CALLABLE bool8& operator=(from_type&& rhs) {
    this->value = static_cast<value_type>(static_cast<bool>(std::move(rhs)));
    return *this;
  }

  template <typename from_type>
  CUDA_HOST_DEVICE_CALLABLE bool8& operator=(const from_type& rhs) {
    this->value = static_cast<value_type>(static_cast<bool>(rhs));
    return *this;
  }

  // conversion operators

  CUDA_HOST_DEVICE_CALLABLE constexpr operator bool() const {
    return static_cast<bool>(this->value);
  }

  CUDA_HOST_DEVICE_CALLABLE explicit operator value_type() const {
    return static_cast<value_type>(static_cast<bool>(this->value));
  }

  // ostream << operator overload

  inline std::ostream& operator<<(std::ostream& os) {
    return os << static_cast<bool>(this->value);
  }

  // binary operator overloads

  CUDA_HOST_DEVICE_CALLABLE bool operator==(bool8 const &rhs) const {
    return static_cast<bool>(*this) == static_cast<bool>(rhs);
  }

  CUDA_HOST_DEVICE_CALLABLE bool operator!=(bool8 const &rhs) const {
    return static_cast<bool>(*this) != static_cast<bool>(rhs);
  }

  CUDA_HOST_DEVICE_CALLABLE bool operator<=(bool8 const &rhs) const {
    return static_cast<bool>(*this) <= static_cast<bool>(rhs); 
  }

  CUDA_HOST_DEVICE_CALLABLE bool operator>=(bool8 const &rhs) const {
    return static_cast<bool>(*this) >= static_cast<bool>(rhs); 
  }

  CUDA_HOST_DEVICE_CALLABLE constexpr bool operator<(bool8 const &rhs) const {
    return static_cast<bool>(*this) < static_cast<bool>(rhs);
  }

  CUDA_HOST_DEVICE_CALLABLE bool operator>(bool8 const &rhs) const {
    return static_cast<bool>(*this) > static_cast<bool>(rhs);
  }

  CUDA_HOST_DEVICE_CALLABLE bool8 operator+(bool8 const &rhs) const {
    return static_cast<bool8>(static_cast<bool>(*this) +
                              static_cast<bool>(rhs));
  }

  CUDA_HOST_DEVICE_CALLABLE bool8 operator-(bool8 const &rhs) const {
    return static_cast<bool8>(static_cast<bool>(*this) -
                              static_cast<bool>(rhs));
  }

  CUDA_HOST_DEVICE_CALLABLE bool8 operator*(bool8 const &rhs) const {
    return static_cast<bool8>(static_cast<bool>(*this) &&
                              static_cast<bool>(rhs));
  }

  CUDA_HOST_DEVICE_CALLABLE bool8 operator/(bool8 const &rhs) const {
    return static_cast<bool8>(static_cast<bool>(*this) /
                              static_cast<bool>(rhs));
  }

  // unary operator overloads

  CUDA_HOST_DEVICE_CALLABLE bool8& operator+=(bool8 const &rhs) {
    bool8 &lhs = *this;
    lhs = lhs + rhs;
    return lhs;
  }

  CUDA_HOST_DEVICE_CALLABLE bool8& operator-=(bool8 const &rhs) {
    bool8 &lhs = *this;
    lhs = lhs - rhs;
    return lhs;
  }

  CUDA_HOST_DEVICE_CALLABLE bool8& operator*=(bool8 const &rhs) {
    bool8 &lhs = *this;
    lhs = lhs * rhs;
    return lhs;
  }

  CUDA_HOST_DEVICE_CALLABLE bool8& operator/=(bool8 const &rhs) {
    bool8 &lhs = *this;
    lhs = lhs / rhs;
    return lhs;
  }

private:
  value_type value;
};

// This is necessary for global, constant, non-fundamental types
// We can't rely on --expt-relaxed-constexpr here because `bool8` is not a
// scalar type. See CUDA Programming guide
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-variables
#ifdef __CUDA_ARCH__
__device__ __constant__ static bool8 true_v{bool8::value_type{1}};
__device__ __constant__ static bool8 false_v{bool8::value_type{0}};
#else
static constexpr bool8 true_v{bool8::value_type{1}};
static constexpr bool8 false_v{bool8::value_type{0}};
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

template <>
struct is_integral<cudf::experimental::bool8> : true_type {};

template <>
struct is_arithmetic<cudf::experimental::bool8> : true_type {};

} // std

namespace cub {

template <> struct NumericTraits<cudf::experimental::bool8> :
  BaseTraits<SIGNED_INTEGER, true, false, uint8_t, int8_t> {};

} // cub
