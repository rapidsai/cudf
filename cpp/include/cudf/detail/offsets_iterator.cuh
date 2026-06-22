/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/normalizing_iterator.cuh>
#include <cudf/types.hpp>

namespace cudf {
namespace detail {

/**
 * @brief The offsets normalizing input iterator
 *
 * This is an iterator that can be used for offsets where the underlying
 * type may be int32_t or int64_t.
 *
 * Use the offsetalator_factory to create an appropriate input iterator
 * from an offsets column_view.
 */
struct input_offsetalator : base_normalator<input_offsetalator, int64_t> {
  friend struct base_normalator<input_offsetalator, int64_t>;  // for CRTP

  using reference = int64_t const;  // this keeps STL and thrust happy

  input_offsetalator()                                     = default;
  input_offsetalator(input_offsetalator const&)            = default;
  input_offsetalator(input_offsetalator&&)                 = default;
  input_offsetalator& operator=(input_offsetalator const&) = default;
  input_offsetalator& operator=(input_offsetalator&&)      = default;

  /**
   * @brief Indirection operator returns the value at the current iterator position
   */
  __device__ inline int64_t operator*() const { return operator[](0); }

  /**
   * @brief Array subscript operator returns a value at the input
   * `idx` position as a int64_t value.
   */
  __device__ inline int64_t operator[](size_type idx) const
  {
    void const* tp = p_ + (static_cast<int64_t>(idx) * this->width_);
    return this->width_ == sizeof(int32_t) ? static_cast<int64_t>(*static_cast<int32_t const*>(tp))
                                           : *static_cast<int64_t const*>(tp);
  }

  /**
   * @brief Create an input index normalizing iterator.
   *
   * Use the indexalator_factory to create an iterator instance.
   *
   * @param data   Pointer to an integer array in device memory
   * @param dtype  Type of data in data
   * @param offset Index value within `offsets` to use as the beginning of the iterator
   */
  CUDF_HOST_DEVICE input_offsetalator(void const* data, data_type dtype, size_type offset = 0)
    : base_normalator<input_offsetalator, int64_t>(
        dtype, dtype.id() == type_id::INT32 ? sizeof(int32_t) : sizeof(int64_t)),
      p_{static_cast<char const*>(data)}
  {
#ifndef __CUDA_ARCH__
    CUDF_EXPECTS(dtype.id() == type_id::INT32 || dtype.id() == type_id::INT64,
                 "Unexpected offsets type");
#else
    cudf_assert((dtype.id() == type_id::INT32 || dtype.id() == type_id::INT64) &&
                "Unexpected offsets type");
#endif
    p_ += (this->width_ * static_cast<int64_t>(offset));
  }

 protected:
  char const* p_;  /// pointer to the integer data in device memory
};

/**
 * @brief The offsets normalizing output iterator
 *
 * This is an iterator that can be used for storing offsets values
 * where the underlying type may be either int32_t or int64_t.
 *
 * Use the offsetalator_factory to create an appropriate output iterator
 * from a mutable_column_view.
 *
 */
struct output_offsetalator : base_normalator<output_offsetalator, int64_t> {
  friend struct base_normalator<output_offsetalator, int64_t>;  // for CRTP

  using reference = output_offsetalator const&;  // required for output iterators

  output_offsetalator()                                      = default;
  output_offsetalator(output_offsetalator const&)            = default;
  output_offsetalator(output_offsetalator&&)                 = default;
  output_offsetalator& operator=(output_offsetalator const&) = default;
  output_offsetalator& operator=(output_offsetalator&&)      = default;

  /**
   * @brief Indirection operator returns this iterator instance in order
   * to capture the `operator=(int64)` calls.
   */
  __device__ inline output_offsetalator const& operator*() const { return *this; }

  /**
   * @brief Array subscript operator returns an iterator instance at the specified `idx` position.
   *
   * This allows capturing the subsequent `operator=(int64)` call in this class.
   */
  __device__ inline output_offsetalator const operator[](size_type idx) const
  {
    output_offsetalator tmp{*this};
    tmp.p_ += (static_cast<int64_t>(idx) * this->width_);
    return tmp;
  }

  /**
   * @brief Assign an offset value to the current iterator position
   */
  __device__ inline output_offsetalator const& operator=(int64_t const value) const
  {
    void* tp = p_;
    if (this->width_ == sizeof(int32_t)) {
      (*static_cast<int32_t*>(tp)) = static_cast<int32_t>(value);
    } else {
      (*static_cast<int64_t*>(tp)) = value;
    }
    return *this;
  }

  /**
   * @brief Create an output offsets iterator
   *
   * @param data      Pointer to an integer array in device memory.
   * @param dtype Type of data in data
   */
  CUDF_HOST_DEVICE output_offsetalator(void* data, data_type dtype)
    : base_normalator<output_offsetalator, int64_t>(
        dtype, dtype.id() == type_id::INT32 ? sizeof(int32_t) : sizeof(int64_t)),
      p_{static_cast<char*>(data)}
  {
#ifndef __CUDA_ARCH__
    CUDF_EXPECTS(dtype.id() == type_id::INT32 || dtype.id() == type_id::INT64,
                 "Unexpected offsets type");
#else
    cudf_assert((dtype.id() == type_id::INT32 || dtype.id() == type_id::INT64) &&
                "Unexpected offsets type");
#endif
  }

 protected:
  char* p_;  /// pointer to the integer data in device memory
};

}  // namespace detail
}  // namespace cudf
