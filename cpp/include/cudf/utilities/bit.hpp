/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cassert>
#include <climits>
#include <cudf/types.hpp>

/**
 * @file bit.hpp
 * @brief Utilities for bit and bitmask operations.
 */

namespace cudf {
namespace detail {
// Work around a bug in NVRTC that fails to compile assert() in constexpr
// functions (fixed after CUDA 11.0)
#if defined __GNUC__
#define LIKELY(EXPR) __builtin_expect(!!(EXPR), 1)
#else
#define LIKELY(EXPR) (!!(EXPR))
#endif

#ifdef NDEBUG
#define constexpr_assert(CHECK) static_cast<void>(0)
#else
#define constexpr_assert(CHECK) (LIKELY(CHECK) ? void(0) : [] { assert(!#CHECK); }())
#endif

template <typename T>
constexpr CUDA_HOST_DEVICE_CALLABLE std::size_t size_in_bits()
{
  static_assert(CHAR_BIT == 8, "Size of a byte must be 8 bits.");
  return sizeof(T) * CHAR_BIT;
}
}  // namespace detail

/**
 * @addtogroup utility_bitmask
 * @{
 * @file
 */

/**
 * @brief Returns the index of the word containing the specified bit.
 */
constexpr CUDA_HOST_DEVICE_CALLABLE size_type word_index(size_type bit_index)
{
  return bit_index / detail::size_in_bits<bitmask_type>();
}

/**
 * @brief Returns the position within a word of the specified bit.
 */
constexpr CUDA_HOST_DEVICE_CALLABLE size_type intra_word_index(size_type bit_index)
{
  return bit_index % detail::size_in_bits<bitmask_type>();
}

/**
 * @brief Sets the specified bit to `1`
 *
 * This function is not thread-safe, i.e., attempting to update bits within the
 * same word concurrently from multiple threads results in undefined behavior.
 *
 * @param bitmask The bitmask containing the bit to set
 * @param bit_index Index of the bit to set
 */
CUDA_HOST_DEVICE_CALLABLE void set_bit_unsafe(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  bitmask[word_index(bit_index)] |= (bitmask_type{1} << intra_word_index(bit_index));
}

/**
 * @brief Sets the specified bit to `0`
 *
 * This function is not thread-safe, i.e., attempting to update bits within the
 * same word concurrently from multiple threads results in undefined behavior.
 *
 * @param bitmask The bitmask containing the bit to clear
 * @param bit_index The index of the bit to clear
 */
CUDA_HOST_DEVICE_CALLABLE void clear_bit_unsafe(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  bitmask[word_index(bit_index)] &= ~(bitmask_type{1} << intra_word_index(bit_index));
}

/**
 * @brief Indicates whether the specified bit is set to `1`
 *
 * @param bit_index Index of the bit to test
 * @return true The specified bit is `1`
 * @return false  The specified bit is `0`
 */
CUDA_HOST_DEVICE_CALLABLE bool bit_is_set(bitmask_type const* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  return bitmask[word_index(bit_index)] & (bitmask_type{1} << intra_word_index(bit_index));
}

/**
 * @brief Returns a bitmask word with the `n` least significant bits set.
 *
 * Behavior is undefined if `n < 0` or if `n >= size_in_bits<bitmask_type>()`
 *
 * @param n The number of least significant bits to set
 * @return A bitmask word with `n` least significant bits set
 */
constexpr CUDA_HOST_DEVICE_CALLABLE bitmask_type set_least_significant_bits(size_type n)
{
  constexpr_assert(0 <= n && n < static_cast<size_type>(detail::size_in_bits<bitmask_type>()));
  return ((bitmask_type{1} << n) - 1);
}

/**
 * @brief Returns a bitmask word with the `n` most significant bits set.
 *
 * Behavior is undefined if `n < 0` or if `n >= size_in_bits<bitmask_type>()`
 *
 * @param n The number of most significant bits to set
 * @return A bitmask word with `n` most significant bits set
 */
constexpr CUDA_HOST_DEVICE_CALLABLE bitmask_type set_most_significant_bits(size_type n)
{
  constexpr size_type word_size{detail::size_in_bits<bitmask_type>()};
  constexpr_assert(0 <= n && n < word_size);
  return ~((bitmask_type{1} << (word_size - n)) - 1);
}

#ifdef __CUDACC__

/**
 * @brief Sets the specified bit to `1`
 *
 * @note This operation requires a global atomic operation. Therefore, it is
 * not recommended to use this function in performance critical regions. When
 * possible, it is more efficient to compute and update an entire word at
 * once using `set_word`.
 *
 * This function is thread-safe.
 *
 * @param bitmask The bitmask containing the bit to set
 * @param bit_index  Index of the bit to set
 */
__device__ inline void set_bit(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  atomicOr(&bitmask[word_index(bit_index)], (bitmask_type{1} << intra_word_index(bit_index)));
}

/**
 * @brief Sets the specified bit to `0`
 *
 * @note This operation requires a global atomic operation. Therefore, it is
 * not recommended to use this function in performance critical regions. When
 * possible, it is more efficient to compute and update an entire element at
 * once using `set_element`.

 * This function is thread-safe.
 *
 * @param bit_index  Index of the bit to clear
 */
__device__ inline void clear_bit(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  atomicAnd(&bitmask[word_index(bit_index)], ~(bitmask_type{1} << intra_word_index(bit_index)));
}
#endif
/** @} */  // end of group
}  // namespace cudf
