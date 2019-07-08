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

#include <cassert>
#include <climits>

namespace cudf {

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
   * @brief Returns the number of represented bits.
   *---------------------------------------------------------------------------**/
  size_type size() const noexcept { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the bit index offset of the first represented bit in the
   * mask
   *---------------------------------------------------------------------------**/
  size_type offset() const noexcept { return _offset; }

  /**---------------------------------------------------------------------------*
   * @brief Return raw pointer to the mask's device memory
   *---------------------------------------------------------------------------**/
  bitmask_type* data() noexcept { return _mask; }

  /**---------------------------------------------------------------------------*
   * @brief Return raw pointer to the mask's device memory
   *---------------------------------------------------------------------------**/
  bitmask_type const* data() const noexcept { return _mask; }

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
   * @brief Returns the number of represented bits.
   *---------------------------------------------------------------------------**/
  size_type size() const noexcept { return mutable_view.size(); }

  /**---------------------------------------------------------------------------*
   * @brief Returns the bit index offset of the first represented bit in the
   * mask
   *---------------------------------------------------------------------------**/
  size_type offset() const noexcept { return mutable_view.offset(); }

  /**---------------------------------------------------------------------------*
   * @brief Return raw pointer to the mask's device memory
   *---------------------------------------------------------------------------**/
  bitmask_type const* data() const noexcept { return mutable_view.data(); }

 private:
  mutable_bitmask_view const mutable_view{};
};

}  // namespace cudf