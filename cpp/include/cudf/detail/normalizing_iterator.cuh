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
struct base_normalator {
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
    template <typename T, std::enable_if_t<not cudf::is_index_type<T>()>* = nullptr>
    CUDF_HOST_DEVICE constexpr std::size_t operator()() const
    {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("only integral types are supported");
#else
      CUDF_UNREACHABLE("only integral types are supported");
#endif
    }
    template <typename T, std::enable_if_t<cudf::is_index_type<T>()>* = nullptr>
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

  int32_t width_;    /// integer type width = 1,2,4, or 8
  data_type dtype_;  /// for type-dispatcher calls
};

/**
 * @brief The integer normalizing input iterator
 *
 * This is an iterator that can be used for index types (integers) without
 * requiring a type-specific instance. It can be used for any iterator
 * interface for reading an array of integer values of type
 * int8, int16, int32, int64, uint8, uint16, uint32, or uint64.
 * Reading specific elements always return a type of `Integer`
 *
 * @tparam Integer Type returned by all read functions
 */
template <typename Integer>
struct input_normalator : base_normalator<input_normalator<Integer>, Integer> {
  friend struct base_normalator<input_normalator<Integer>, Integer>;  // for CRTP

  using reference = Integer const;  // this keeps STL and thrust happy

  input_normalator()                                   = default;
  input_normalator(input_normalator const&)            = default;
  input_normalator(input_normalator&&)                 = default;
  input_normalator& operator=(input_normalator const&) = default;
  input_normalator& operator=(input_normalator&&)      = default;

  /**
   * @brief Indirection operator returns the value at the current iterator position
   */
  __device__ inline Integer operator*() const { return operator[](0); }

  /**
   * @brief Dispatch functor for resolving a Integer value from any integer type
   */
  struct normalize_type {
    template <typename T, std::enable_if_t<cudf::is_index_type<T>()>* = nullptr>
    __device__ Integer operator()(void const* tp)
    {
      return static_cast<Integer>(*static_cast<T const*>(tp));
    }
    template <typename T, std::enable_if_t<not cudf::is_index_type<T>()>* = nullptr>
    __device__ Integer operator()(void const*)
    {
      CUDF_UNREACHABLE("only integral types are supported");
    }
  };

  /**
   * @brief Array subscript operator returns a value at the input
   * `idx` position as a `Integer` value.
   */
  __device__ inline Integer operator[](size_type idx) const
  {
    void const* tp = p_ + (idx * this->width_);
    return type_dispatcher(this->dtype_, normalize_type{}, tp);
  }

  /**
   * @brief Create an input index normalizing iterator.
   *
   * Use the indexalator_factory to create an iterator instance.
   *
   * @param data      Pointer to an integer array in device memory.
   * @param data_type Type of data in data
   */
  CUDF_HOST_DEVICE input_normalator(void const* data, data_type dtype, cudf::size_type offset = 0)
    : base_normalator<input_normalator<Integer>, Integer>(dtype), p_{static_cast<char const*>(data)}
  {
    p_ += offset * this->width_;
  }

  char const* p_;  /// pointer to the integer data in device memory
};

/**
 * @brief The integer normalizing output iterator
 *
 * This is an iterator that can be used for index types (integers) without
 * requiring a type-specific instance. It can be used for any iterator
 * interface for writing an array of integer values of type
 * int8, int16, int32, int64, uint8, uint16, uint32, or uint64.
 * Setting specific elements always accept the `Integer` type values.
 *
 * @tparam Integer The type used for all write functions
 */
template <typename Integer>
struct output_normalator : base_normalator<output_normalator<Integer>, Integer> {
  friend struct base_normalator<output_normalator<Integer>, Integer>;  // for CRTP

  using reference = output_normalator const&;  // required for output iterators

  output_normalator()                                    = default;
  output_normalator(output_normalator const&)            = default;
  output_normalator(output_normalator&&)                 = default;
  output_normalator& operator=(output_normalator const&) = default;
  output_normalator& operator=(output_normalator&&)      = default;

  /**
   * @brief Indirection operator returns this iterator instance in order
   * to capture the `operator=(Integer)` calls.
   */
  __device__ inline output_normalator const& operator*() const { return *this; }

  /**
   * @brief Array subscript operator returns an iterator instance at the specified `idx` position.
   *
   * This allows capturing the subsequent `operator=(Integer)` call in this class.
   */
  __device__ inline output_normalator const operator[](size_type idx) const
  {
    output_normalator tmp{*this};
    tmp.p_ += (idx * this->width_);
    return tmp;
  }

  /**
   * @brief Dispatch functor for setting the index value from a size_type value.
   */
  struct normalize_type {
    template <typename T, std::enable_if_t<cudf::is_index_type<T>()>* = nullptr>
    __device__ void operator()(void* tp, Integer const value)
    {
      (*static_cast<T*>(tp)) = static_cast<T>(value);
    }
    template <typename T, std::enable_if_t<not cudf::is_index_type<T>()>* = nullptr>
    __device__ void operator()(void*, Integer const)
    {
      CUDF_UNREACHABLE("only index types are supported");
    }
  };

  /**
   * @brief Assign an Integer value to the current iterator position
   */
  __device__ inline output_normalator const& operator=(Integer const value) const
  {
    void* tp = p_;
    type_dispatcher(this->dtype_, normalize_type{}, tp, value);
    return *this;
  }

  /**
   * @brief Create an output normalizing iterator
   *
   * @param data      Pointer to an integer array in device memory.
   * @param data_type Type of data in data
   */
  CUDF_HOST_DEVICE output_normalator(void* data, data_type dtype)
    : base_normalator<output_normalator<Integer>, Integer>(dtype), p_{static_cast<char*>(data)}
  {
  }

  char* p_;  /// pointer to the integer data in device memory
};

}  // namespace detail
}  // namespace cudf
