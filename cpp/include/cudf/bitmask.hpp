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
#include "types.hpp"

#include <rmm/device_buffer.hpp>

#include <cassert>
#include <climits>

// Forward decls
namespace rmm {
class device_memory_resource;
device_memory_resource* get_default_resource();
}  // namespace rmm

namespace cudf {

namespace detail {
template <typename T>
constexpr inline std::size_t size_in_bits() {
  return sizeof(T) * CHAR_BIT;
}

constexpr __host__ __device__ size_type element_index(size_type bit_index) {
  return bit_index / size_in_bits<bitmask_type>();
}

constexpr __host__ __device__ size_type
intra_element_index(size_type bit_index) {
  return bit_index % size_in_bits<bitmask_type>();
}
}  // namespace detail

class mutable_bitmask_view {
 public:
  mutable_bitmask_view(bitmask_type* mask, size_type size)
      : _mask{mask}, _size{size} {}

  __device__ bool is_valid(size_type bit_index) const noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    // TODO Implement
    return true;
  }

  __device__ bool is_null(size_type bit_index) const noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    return not is_valid(bit_index);
  }

  __device__ void set_valid(size_type bit_index) noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    // TODO Implement
    return;
  }

  __device__ void set_null(size_type bit_index) noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    // TODO Implement
    return;
  }

  __device__ bitmask_type get_element(size_type element_index) const noexcept {
    assert(element_index >= 0);
    // TODO Implement
    return 0xffffffff;
  }
  __device__ void set_element(size_type element_index) noexcept {
    assert(element_index >= 0);
    // TODO Implement
    return;
  }

  __host__ __device__ bitmask_type* data() noexcept { return _mask; }

  __host__ __device__ bitmask_type const* data() const noexcept {
    return _mask;
  }

 private:
  bitmask_type* _mask{nullptr};
  size_type _size{0};
};

class bitmask_view {
 public:
  bitmask_view(bitmask_type const* mask, size_type size)
      : mutable_view{const_cast<bitmask_type*>(mask), size} {}

  bitmask_view(mutable_bitmask_view m_view) : mutable_view{m_view} {}

  __device__ bool is_valid(size_type bit_index) const noexcept {
    return mutable_view.is_valid(bit_index);
  }

  __device__ bool is_null(size_type bit_index) const noexcept {
    return mutable_view.is_null(bit_index);
  }

  __device__ bitmask_type get_element(size_type element_index) const noexcept {
    return mutable_view.get_element(element_index);
  }

  __host__ __device__ bitmask_type const* data() const noexcept {
    return mutable_view.data();
  }

 private:
  mutable_bitmask_view const mutable_view;
};

class bitmask {
 public:
  bitmask() = default;
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
   * @param padding_boundary[in]  optional, specifies the quantum, in bytes, of
   * the amount of memory allocated (i.e. the allocation size is padded to a
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
   * Requires that `buffer` contain sufficient storage to represent `size` bits.
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
   * Requires that `buffer` contain sufficient storage to represent `size` bits.
   *
   * @param size The number of bits represented by the bitmask
   * @param mask The `device_buffer` to move from
   *---------------------------------------------------------------------------**/
  bitmask(size_type size, rmm::device_buffer&& mask) noexcept;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new bitmask by copying from an existing `bitmask_view`.
   *
   * @param view[in]  The `bitmask_view` to copy from.
   * @param mr[in] optional, the `device_memory_resource` to use for device
   * memory allocation
   *---------------------------------------------------------------------------**/
  bitmask(bitmask_view view, rmm::mr::device_memory_resource* mr =
                                 rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Construct a new bitmask by copying from an existing
   * `mutable_bitmask_view`
   *
   * @param view The `mutable_bitmask_view` to copy from.
   * @param mr[in] optional, the `device_memory_resource` to use for device
   * memory allocation
   *---------------------------------------------------------------------------**/
  bitmask(mutable_bitmask_view view, rmm::mr::device_memory_resource* mr =
                                         rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of bits represented by the bitmask.
   *---------------------------------------------------------------------------**/
  size_type size() const noexcept { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Constructs an immutable, zero-copy `bitmask_view` with device-level
   * accessors for the contents of the bitmask.
   *
   * @return bitmask_view The view of the bitmask data
   *---------------------------------------------------------------------------**/
  bitmask_view view() const noexcept { return this->operator bitmask_view(); }

  /**---------------------------------------------------------------------------*
   * @brief Constructs a mutable, zero-copy view of the bitmask with
   * device-level accessors and functions for modifying the contents of the
   * bitmask.
   *
   * @return mutable_bitmask_view The mutable view of the bitmask data
   *---------------------------------------------------------------------------**/
  mutable_bitmask_view mutable_view() noexcept {
    return this->operator mutable_bitmask_view();
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a zero-copy `mutable_bitmask_view` from this `bitmask`
   *---------------------------------------------------------------------------**/
  operator mutable_bitmask_view() noexcept {
    return mutable_bitmask_view{static_cast<bitmask_type*>(_data.data()),
                                _size};
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a zero-copy immutable `bitmask_view` from this `bitmask`.
   *---------------------------------------------------------------------------**/
  operator bitmask_view() const noexcept {
    return bitmask_view{static_cast<bitmask_type const*>(_data.data()), _size};
  }

 private:
  rmm::device_buffer _data{};
  size_type _size{};
};

}  // namespace cudf