/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/utilities/type_dispatcher.hpp>

#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief The base class for the input or output normalizing iterator
 *
 * The base class mainly manages updating the `p_` member variable while the
 * subclasses handle accessing individual elements in device memory.
 *
 * @tparam Derived The derived class type for the iterator
 * @tparam Integer The type the iterator normalizes to
 */
template <class Derived, typename Integer>
struct alignas(16) base_normalator {
  static_assert(cudf::is_index_type<Integer>());
  using difference_type   = std::ptrdiff_t;
  using value_type        = Integer;
  using pointer           = Integer*;
  using iterator_category = std::random_access_iterator_tag;

  base_normalator()                                  = default;
  base_normalator(base_normalator const&)            = default;
  base_normalator(base_normalator&&)                 = default;
  base_normalator& operator=(base_normalator const&) = default;
  base_normalator& operator=(base_normalator&&)      = default;

  /**
   * @brief Prefix increment operator.
   */
  CUDF_HOST_DEVICE inline Derived& operator++()
  {
    Derived& derived = static_cast<Derived&>(*this);
    derived.p_ += width_;
    return derived;
  }

  /**
   * @brief Postfix increment operator.
   */
  CUDF_HOST_DEVICE inline Derived operator++(int)
  {
    Derived tmp{static_cast<Derived&>(*this)};
    operator++();
    return tmp;
  }

  /**
   * @brief Prefix decrement operator.
   */
  CUDF_HOST_DEVICE inline Derived& operator--()
  {
    Derived& derived = static_cast<Derived&>(*this);
    derived.p_ -= width_;
    return derived;
  }

  /**
   * @brief Postfix decrement operator.
   */
  CUDF_HOST_DEVICE inline Derived operator--(int)
  {
    Derived tmp{static_cast<Derived&>(*this)};
    operator--();
    return tmp;
  }

  /**
   * @brief Compound assignment by sum operator.
   */
  CUDF_HOST_DEVICE inline Derived& operator+=(difference_type offset)
  {
    Derived& derived = static_cast<Derived&>(*this);
    derived.p_ += offset * width_;
    return derived;
  }

  /**
   * @brief Increment by offset operator.
   */
  CUDF_HOST_DEVICE inline Derived operator+(difference_type offset) const
  {
    auto tmp = Derived{static_cast<Derived const&>(*this)};
    tmp.p_ += (offset * width_);
    return tmp;
  }

  /**
   * @brief Addition assignment operator.
   */
  CUDF_HOST_DEVICE inline friend Derived operator+(difference_type offset, Derived const& rhs)
  {
    Derived tmp{rhs};
    tmp.p_ += (offset * rhs.width_);
    return tmp;
  }

  /**
   * @brief Compound assignment by difference operator.
   */
  CUDF_HOST_DEVICE inline Derived& operator-=(difference_type offset)
  {
    Derived& derived = static_cast<Derived&>(*this);
    derived.p_ -= offset * width_;
    return derived;
  }

  /**
   * @brief Decrement by offset operator.
   */
  CUDF_HOST_DEVICE inline Derived operator-(difference_type offset) const
  {
    auto tmp = Derived{static_cast<Derived const&>(*this)};
    tmp.p_ -= (offset * width_);
    return tmp;
  }

  /**
   * @brief Subtraction assignment operator.
   */
  CUDF_HOST_DEVICE inline friend Derived operator-(difference_type offset, Derived const& rhs)
  {
    Derived tmp{rhs};
    tmp.p_ -= (offset * rhs.width_);
    return tmp;
  }

  /**
   * @brief Compute offset from iterator difference operator.
   */
  CUDF_HOST_DEVICE inline difference_type operator-(Derived const& rhs) const
  {
    return (static_cast<Derived const&>(*this).p_ - rhs.p_) / width_;
  }

  /**
   * @brief Equals to operator.
   */
  CUDF_HOST_DEVICE inline bool operator==(Derived const& rhs) const
  {
    return rhs.p_ == static_cast<Derived const&>(*this).p_;
  }

  /**
   * @brief Not equals to operator.
   */
  CUDF_HOST_DEVICE inline bool operator!=(Derived const& rhs) const
  {
    return rhs.p_ != static_cast<Derived const&>(*this).p_;
  }

  /**
   * @brief Less than operator.
   */
  CUDF_HOST_DEVICE inline bool operator<(Derived const& rhs) const
  {
    return static_cast<Derived const&>(*this).p_ < rhs.p_;
  }

  /**
   * @brief Greater than operator.
   */
  CUDF_HOST_DEVICE inline bool operator>(Derived const& rhs) const
  {
    return static_cast<Derived const&>(*this).p_ > rhs.p_;
  }

  /**
   * @brief Less than or equals to operator.
   */
  CUDF_HOST_DEVICE inline bool operator<=(Derived const& rhs) const
  {
    return static_cast<Derived const&>(*this).p_ <= rhs.p_;
  }

  /**
   * @brief Greater than or equals to operator.
   */
  CUDF_HOST_DEVICE inline bool operator>=(Derived const& rhs) const
  {
    return static_cast<Derived const&>(*this).p_ >= rhs.p_;
  }

 private:
  struct integer_sizeof_fn {
    template <typename T, CUDF_ENABLE_IF(not cudf::is_fixed_width<T>())>
    CUDF_HOST_DEVICE constexpr std::size_t operator()() const
    {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("only integral types are supported");
#else
      CUDF_UNREACHABLE("only integral types are supported");
#endif
    }
    template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
    CUDF_HOST_DEVICE constexpr std::size_t operator()() const noexcept
    {
      return sizeof(T);
    }
  };

 protected:
  /**
   * @brief Constructor assigns width and type member variables for base class.
   */
  explicit CUDF_HOST_DEVICE base_normalator(data_type dtype) : dtype_(dtype)
  {
    width_ = static_cast<int32_t>(type_dispatcher(dtype, integer_sizeof_fn{}));
  }

  /**
   * @brief Constructor assigns width and type member variables for base class.
   */
  explicit CUDF_HOST_DEVICE base_normalator(data_type dtype, int32_t width)
    : width_(width), dtype_(dtype)
  {
  }

  int32_t width_;    /// integer type width = 1,2,4, or 8
  data_type dtype_;  /// for type-dispatcher calls
};

}  // namespace detail
}  // namespace cudf
