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

#include <cassert>
#include <climits>

namespace cudf {

namespace detail {
template <typename T>
constexpr inline std::size_t size_in_bits() {
  return sizeof(T) * CHAR_BIT;
}

constexpr __host__ __device__ inline size_type element_index(
    size_type bit_index) {
  return bit_index / size_in_bits<bitmask_type>();
}

constexpr __host__ __device__ inline size_type intra_element_index(
    size_type bit_index) {
  return bit_index % size_in_bits<bitmask_type>();
}
}  // namespace detail

class mutable_bitmask_view {
 public:
  mutable_bitmask_view() = default;
  ~mutable_bitmask_view() = default;
  mutable_bitmask_view(mutable_bitmask_view const& other) = default;
  mutable_bitmask_view(mutable_bitmask_view&& other) = default;
  mutable_bitmask_view& operator=(mutable_bitmask_view const& other) = default;
  mutable_bitmask_view& operator=(mutable_bitmask_view&& other) = default;

  mutable_bitmask_view(bitmask_type* mask, size_type size)
      : _mask{mask}, _size{size} {}

  __device__ bool is_valid(size_type bit_index) const noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    auto element_index = detail::element_index(bit_index);
    auto intra_element_index = detail::intra_element_index(bit_index);
    return _mask[element_index] & (bitmask_type{1} << intra_element_index);
  }

  __device__ bool is_null(size_type bit_index) const noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    return not is_valid(bit_index);
  }

  __device__ void set_valid(size_type bit_index) noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    auto element_index = detail::element_index(bit_index);
    auto intra_element_index = detail::intra_element_index(bit_index);
    atomicOr(&_mask[element_index], (bitmask_type{1} << intra_element_index));
  }

  __device__ void set_null(size_type bit_index) noexcept {
    assert(bit_index >= 0);
    assert(bit_index < _size);
    auto element_index = detail::element_index(bit_index);
    auto intra_element_index = detail::intra_element_index(bit_index);
    atomicAnd(&_mask[element_index], ~(bitmask_type{1} << intra_element_index));
  }

  __device__ bitmask_type get_element(size_type element_index) const noexcept {
    assert(element_index >= 0);
    auto element_index = detail::element_index(bit_index);
    return _mask[element_index];
  }

  __device__ void set_element(bitmask_type new_element,
                              size_type element_index) noexcept {
    assert(element_index >= 0);
    auto element_index = detail::element_index(bit_index);
    return _mask[element_index] = new_element;
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
  bitmask_view() = default;
  ~bitmask_view() = default;
  bitmask_view(bitmask_view const& other) = default;
  bitmask_view(bitmask_view&& other) = default;
  bitmask_view& operator=(bitmask_view const& other) = default;
  bitmask_view& operator=(bitmask_view&& other) = default;

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
  mutable_bitmask_view const mutable_view{};
};

}  // namespace cudf