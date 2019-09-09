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

#include <cudf/types.hpp>

#include <cassert>

namespace cudf {

namespace detail {
template <typename T>
constexpr inline std::size_t size_in_bits() {
  static_assert(CHAR_BIT == 8, "Size of a byte must be 8 bits.");
  return sizeof(T) * CHAR_BIT;
}
}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief Returns the index of the element containing the specified bit.
 *---------------------------------------------------------------------------**/
constexpr __host__ __device__ inline size_type element_index(
    size_type bit_index) {
  return bit_index / detail::size_in_bits<bitmask_type>();
}

/**---------------------------------------------------------------------------*
 * @brief Returns the position within an element of the specified bit.
 *---------------------------------------------------------------------------**/
constexpr __host__ __device__ inline size_type intra_element_index(
    size_type bit_index) {
  return bit_index % detail::size_in_bits<bitmask_type>();
}

/**---------------------------------------------------------------------------*
 * @brief Indicates if the specified bit is set to `1`
 *
 * @param bit_index Index of the bit to test
 * @return true The specified bit is `1`
 * @return false  The specified bit is `0`
 *---------------------------------------------------------------------------**/
__device__ inline bool bit_is_set(bitmask_type const* bitmask, size_type bit_index) {
  assert(nullptr != bitmask);
  return bitmask[element_index(bit_index)] &
         (bitmask_type{1} << intra_element_index(bit_index));
}

/**---------------------------------------------------------------------------*
 * @brief Sets the specified bit to `1`
 *
 * @note This operation requires a global atomic operation. Therefore, it is
 * not reccomended to use this function in performance critical regions. When
 * possible, it is more efficient to compute and update an entire element at
 * once using `set_element`.
 *
 * This function is thread-safe.
 *
 * @param bit_index  Index of the bit to set
 *---------------------------------------------------------------------------**/
__device__ inline void set_bit(bitmask_type* bitmask, size_type bit_index) {
  assert(nullptr != bitmask);
  atomicOr(&bitmask[element_index(bit_index)],
           (bitmask_type{1} << intra_element_index(bit_index)));
}

/**---------------------------------------------------------------------------*
 * @brief Sets the specified bit to `0`
 *
 * @note This operation requires a global atomic operation. Therefore, it is
 * not reccomended to use this function in performance critical regions. When
 * possible, it is more efficient to compute and update an entire element at
 * once using `set_element`.

 * This function is thread-safe.
 *
 * @param bit_index  Index of the bit to set
 *---------------------------------------------------------------------------**/
__device__ inline void clear_bit(bitmask_type* bitmask, size_type bit_index) {
  assert(nullptr != bitmask);
  atomicAnd(&bitmask[element_index(bit_index)],
            ~(bitmask_type{1} << intra_element_index(bit_index)));
}

}  // namespace cudf