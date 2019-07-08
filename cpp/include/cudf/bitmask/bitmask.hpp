/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#include "bitmask_view.hpp"

#include <rmm/device_buffer.hpp>

#include <cassert>
#include <climits>

// Forward decls
namespace rmm {
class device_memory_resource;
device_memory_resource* get_default_resource();
}  // namespace rmm

namespace cudf {
/**---------------------------------------------------------------------------*
 * @brief A memory owning bitmask class.
 *
 *  A `bitmask` is a contiguous set of `size` bits in device memory. The bits in
 * the range `[0, size)` are the "represented bits". Bits outside this range are
 * undefined.
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
class bitmask {
 public:
  bitmask() = default;
  ~bitmask() = default;
  bitmask(bitmask const& other) = default;
  bitmask(bitmask&& other) = default;
  bitmask& operator=(bitmask const& other) = delete;
  bitmask& operator=(bitmask&& other) = delete;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new bitmask with a sufficiently sized device memory
   * allocation to represent `size` bits.
   *
   * @note Bits outside the range [0,size) are undefined.
   *
   * @param size[in] The minimum number of bits in the bitmask
   * @param padding_boundary[in]  optional, specifies the quantum, in bytes,
   *of the amount of memory allocated (i.e. the allocation size is padded to a
   * multiple of this value).
   * @param mr[in] optional, the `device_memory_resource` to use for device
   * memory allocation
   *---------------------------------------------------------------------------**/
  explicit bitmask(
      size_type size, size_type padding_boundary = 64,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Construct a new bitmask by copying from an existing device_buffer.
   *
   * Copies the contents of a `device_buffer` to use a bitmask.
   *
   * Requires that `buffer` contain sufficient storage to represent `size`
   *bits.
   *
   * @param size The number of bits represented by the bitmask
   * @param buffer The `device_buffer` to be copied from
   *---------------------------------------------------------------------------**/
  bitmask(size_type size, rmm::device_buffer const& buffer);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new bitmask by moving from an existing device_buffer.
   *
   * Moves the contents from a `device_buffer` to use a bitmask.
   *
   * Requires that `buffer` contain sufficient storage to represent `size`
   *bits.
   *
   * @param size The number of bits represented by the bitmask
   * @param mask The `device_buffer` to move from
   *---------------------------------------------------------------------------**/
  bitmask(size_type size, rmm::device_buffer&& mask) noexcept;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new bitmask by copying from an existing
   *`bitmask_view`.
   *
   * @param view[in]  The `bitmask_view` to copy from.
   * @param mr[in] optional, the `device_memory_resource` to use for device
   * memory allocation
   *---------------------------------------------------------------------------**/
  explicit bitmask(bitmask_view view, rmm::mr::device_memory_resource* mr =
                                          rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Construct a new bitmask by copying from an existing
   * `mutable_bitmask_view`
   *
   * @param view The `mutable_bitmask_view` to copy from.
   * @param mr[in] optional, the `device_memory_resource` to use for device
   * memory allocation
   *---------------------------------------------------------------------------**/
  explicit bitmask(
      mutable_bitmask_view view,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of bits represented by the bitmask.
   *---------------------------------------------------------------------------**/
  size_type size() const noexcept { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Constructs an immutable, zero-copy view of the bitmask
   *
   * @return bitmask_view The view of the bitmask data
   *---------------------------------------------------------------------------**/
  bitmask_view view() const noexcept;

  /**---------------------------------------------------------------------------*
   * @brief Constructs a mutable, zero-copy view of the bitmask
   *
   * @return mutable_bitmask_view The mutable view of the bitmask data
   *---------------------------------------------------------------------------**/
  mutable_bitmask_view mutable_view() noexcept;

  /**---------------------------------------------------------------------------*
   * @brief Constructs a zero-copy immutable view of a "slice" of the bitmask
   * with the specified offset and size.
   *
   * @param offset The bit index of the first bit in the slice
   * @param size optional, the number of bits in the slice. If zero, slices from
   * the offset to the size of the source bitmask
   * @return bitmask_view
   *---------------------------------------------------------------------------**/
  bitmask_view slice(size_type offset, size_type size = 0) const noexcept;

  /**---------------------------------------------------------------------------*
   * @brief Constructs a zero-copy mutable view of a "slice" of the bitmask
   * with the specified offset and size.
   *
   * @param offset The bit index of the first bit in the slice
   * @param size optional, the number of bits in the slice. If zero, slices from
   * the offset to the size of the source bitmask
   * @return bitmask_view
   *---------------------------------------------------------------------------**/
  mutable_bitmask_view mutable_slice(size_type offset, size_type size = 0) const
      noexcept;

  /**---------------------------------------------------------------------------*
   * @brief Construct a zero-copy `mutable_bitmask_view` from this `bitmask`
   *---------------------------------------------------------------------------**/
  operator mutable_bitmask_view() noexcept;

  /**---------------------------------------------------------------------------*
   * @brief Construct a zero-copy immutable `bitmask_view` from this
   *`bitmask`.
   *---------------------------------------------------------------------------**/
  operator bitmask_view() const noexcept;

 private:
  rmm::device_buffer _data{};
  size_type _size{};
};

}  // namespace cudf