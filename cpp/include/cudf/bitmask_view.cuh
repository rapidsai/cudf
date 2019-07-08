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
#include "types.hpp"

#include <cassert>
#include <climits>

namespace cudf {

namespace detail {
template <typename T>
constexpr inline std::size_t size_in_bits() {
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
 * @brief A mutable, non-owning view of a device memory allocation as a
 * "bitmask".
 *
 * A `bitmask` is a contiguous set of `m` `bitmask_type`"element"s. The
 * `size` of a bitmask refers to the number of bits it represents. Each bitmask
 * `view` has an `offset` that indicates the bit index of the first represented
 * bit in the mask (by default, the `offset` is zero).
 *
 * The "represented bits" are the contiguous set of bits in the range `[offset,
 * size+offset)`. The device memory allocation may be larger than what is
 * required to represent `size` bits, and bits outside of the represented bit
 * range are undefined.
 *
 * The `bitmask` uses LSB ordering, e.g., `bit_index` 0 refers to the
 * *least* significant bit of the *first* element in the `bitmask`.
 *
 * For example:
 * ```
 * bit index:  7 6 5 4 3 2 1 0
 * bits: (MSB) 0 1 0 1 1 1 1 1 (LSB)
 * ```
 *
 * Provides `__device__` functions for accessing and modifying the state of
 * elements and individual bits within the bitmask.
 *---------------------------------------------------------------------------**/
class mutable_bitmask_view {
 public:
  mutable_bitmask_view() = default;
  ~mutable_bitmask_view() = default;
  mutable_bitmask_view(mutable_bitmask_view const& other) = default;
  mutable_bitmask_view(mutable_bitmask_view&& other) = default;
  mutable_bitmask_view& operator=(mutable_bitmask_view const& other) = default;
  mutable_bitmask_view& operator=(mutable_bitmask_view&& other) = default;

  /**---------------------------------------------------------------------------*
   * @brief Construct a `mutable_bitmask_view` from a raw device memory pointer
   * and a size.
   *
   * Optionally accepts an `offset` (defaults to zero) to allow zero-copy
   * slicing. The `offset` indicates the bit index of the first represented bit
   * in the mask.
   *
   * Requires that `mask` have 256B or greater power of two alignment.
   *
   * @param mask Pointer to an existing device memory allocation of sufficient
   * size to hold `offset + size` bits.
   * @param size The number of bits represented by the bitmask
   * @param offset optional, the bit index of the first represented bit.
   * Defaults to 0
   *---------------------------------------------------------------------------**/
  mutable_bitmask_view(bitmask_type* mask, size_type size,
                       size_type offset = 0);

  /**---------------------------------------------------------------------------*
   * @brief Indicates if the specified bit is set to `1`
   *
   * @param bit_index Index of the bit to test
   * @return true The specified bit is `1`
   * @return false  The specified bit is `0`
   *---------------------------------------------------------------------------**/
  __device__ bool bit_is_set(size_type bit_index) const noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    return _mask[element_index(bit_index)] &
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
  __device__ void set_bit(size_type bit_index) noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    atomicOr(&_mask[element_index(bit_index)],
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
  __device__ void clear_bit(size_type bit_index) noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    atomicAnd(&_mask[element_index(bit_index)],
              ~(bitmask_type{1} << intra_element_index(bit_index)));
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the element at the specified index
   *
   * @note To find the element index for a particular bit index, use the
   *`element_index(i)` function.
   *
   * @param element_index Index of the requested element
   * @return bitmask_type The requested element
   *---------------------------------------------------------------------------**/
  __device__ bitmask_type get_element(size_type element_index) const noexcept {
    assert(element_index >= 0);
    // TODO Add upper bound check
    return _mask[element_index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Sets the element at the specified index with a new element
   *
   * This function is *not* thread safe, i.e., undefined behavior results if two
   * threads attempt to concurrently read/write an element.
   *
   * @param new_element The new element to be stored at the specified index
   * @param element_index The index of the element to be updated
   *---------------------------------------------------------------------------**/
  __device__ void set_element(bitmask_type new_element,
                              size_type element_index) noexcept {
    assert(element_index >= 0);
    // TODO Add upper bound check
    _mask[element_index] = new_element;
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of represented bits.
   *---------------------------------------------------------------------------**/
  __host__ __device__ size_type size() const noexcept { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the bit index offset of the first represented bit in the
   * mask
   *---------------------------------------------------------------------------**/
  __host__ __device__ size_type offset() const noexcept { return _offset; }

  /**---------------------------------------------------------------------------*
   * @brief Return raw pointer to the mask's device memory
   *---------------------------------------------------------------------------**/
  __host__ __device__ bitmask_type* data() noexcept { return _mask; }

  /**---------------------------------------------------------------------------*
   * @brief Return raw pointer to the mask's device memory
   *---------------------------------------------------------------------------**/
  __host__ __device__ bitmask_type const* data() const noexcept {
    return _mask;
  }

 private:
  bitmask_type* _mask{nullptr};  ///< Pointer to device memory holding the bits
  size_type _size{0};            ///< The number of represented bits
  size_type _offset{0};          ///< Beginning bit index of the bitmask
};

/**---------------------------------------------------------------------------*
 * @brief An immutable, non-owning view of a device memory allocation as a
 * "bitmask".
 *
 * A `bitmask` is a contiguous set of `m` `bitmask_type`"element"s. The
 * `size` of a bitmask refers to the number of bits it represents. Each bitmask
 * `view` has an `offset` that indicates the bit index of the first represented
 * bit in the mask (by default, the `offset` is zero).
 *
 * The "represented bits" are the contiguous set of bits in the range `[offset,
 * size+offset)`. The device memory allocation may be larger than what is
 * required to represent `size` bits, and bits outside of the represented bit
 * range are undefined.
 *
 * The `bitmask` uses LSB ordering, e.g., `bit_index` 0 refers to the
 * *least* significant bit of the *first* element in the `bitmask`.
 *
 * For example:
 * ```
 * bit index:  7 6 5 4 3 2 1 0
 * bits: (MSB) 0 1 0 1 1 1 1 1 (LSB)
 * ```
 *
 * Provides `__device__` functions for accessing the state of
 * elements and individual bits within the bitmask.
 *---------------------------------------------------------------------------**/
class bitmask_view {
 public:
  bitmask_view() = default;
  ~bitmask_view() = default;
  bitmask_view(bitmask_view const& other) = default;
  bitmask_view(bitmask_view&& other) = default;
  bitmask_view& operator=(bitmask_view const& other) = default;
  bitmask_view& operator=(bitmask_view&& other) = default;

  /**---------------------------------------------------------------------------*
   * @brief Construct a `bitmask_view` from a raw device memory pointer
   * and a size.
   *
   * Optionally accepts an `offset` (defaults to zero) to allow zero-copy
   * slicing. The `offset` indicates the bit index of the first represented bit
   * in the mask.
   *
   * Requires that `mask` have 256B or greater power of two alignment.
   *
   * @param mask Pointer to an existing device memory allocation of sufficient
   * size to hold `size` bits.
   * @param size The number of bits represented by the bitmask
   * @param offset optional, the bit index of the first represented bit.
   * Defaults to 0
   *---------------------------------------------------------------------------**/
  bitmask_view(bitmask_type const* mask, size_type size, size_type offset = 0);

  /**---------------------------------------------------------------------------*
   * @brief Construct a `bitmask_view` from a `mutable_bitmask_view`
   *
   * Provides an immutable view from a mutable view.
   *
   * @param m_view The `mutable_bitmask_view` from which to construct an
   * immutable view
   *---------------------------------------------------------------------------**/
  bitmask_view(mutable_bitmask_view m_view);

  /**---------------------------------------------------------------------------*
   * @brief Indicates if the specified bit is set to `1`
   *
   * @param bit_index Index of the bit to test
   * @return true The specified bit is `1`
   * @return false  The specified bit is `0`
   *---------------------------------------------------------------------------**/
  __device__ bool bit_is_set(size_type bit_index) const noexcept {
    return mutable_view.bit_is_set(bit_index);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the element at the specified index
   *
   * @note To find the element index for a particular bit index, use the
   *`element_index(i)` function.
   *
   * @param element_index Index of the requested element
   * @return bitmask_type The requested element
   *---------------------------------------------------------------------------**/
  __device__ bitmask_type get_element(size_type element_index) const noexcept {
    return mutable_view.get_element(element_index);
  }

  /**---------------------------------------------------------------------------*
   * @brief Return raw pointer to the mask's device memory
   *---------------------------------------------------------------------------**/
  __host__ __device__ bitmask_type const* data() const noexcept {
    return mutable_view.data();
  }

 private:
  mutable_bitmask_view const mutable_view{};
};

}  // namespace cudf