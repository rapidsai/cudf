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

#include <cstddef>
#include <limits>
#include <type_traits>

namespace cudf {
namespace detail {

constexpr size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

/**
 * @brief C++20 std::span with reduced feature set.
 */
template <typename T, size_t Extent = dynamic_extent>
class span {
  static_assert(Extent == dynamic_extent, "Only dynamic extent is supported");

  using element_type    = T;
  using value_type      = std::remove_cv<T>;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer         = T*;
  using iterator        = T*;
  using const_pointer   = T const*;
  using reference       = T&;
  using const_reference = T const&;

 public:
  constexpr span() noexcept : _data(nullptr), _size(0) {}
  constexpr span(pointer data, size_type size) : _data(data), _size(size) {}
  // constexpr span(pointer begin, pointer end) : _data(begin), _size(end - begin) {}
  constexpr span(const span& other) noexcept = default;
  constexpr span& operator=(const span& other) noexcept = default;

  // not noexcept due to undefined behavior when size = 0
  constexpr reference front() const { return _data[0]; }
  // not noexcept due to undefined behavior when size = 0
  constexpr reference back() const { return _data[_size - 1]; }
  // not noexcept due to undefined behavior when idx < 0 || idx >= size
  constexpr reference operator[](size_type idx) const { return _data[idx]; }

  constexpr iterator begin() const noexcept { return _data; }
  constexpr iterator end() const noexcept { return _data + _size; }
  constexpr pointer data() const noexcept { return _data; }

  constexpr size_type size() const noexcept { return _size; }
  constexpr size_type size_bytes() const noexcept { return sizeof(T) * _size; }
  constexpr bool empty() const noexcept { return _size == 0; }

  constexpr span<T, Extent> first(size_type count) const noexcept
  {
    return span<T, Extent>(_data, count);
  }

  constexpr span<T, Extent> last(size_type count) const noexcept
  {
    return span<T, Extent>(_data + _size - count, count);
  }

  constexpr span<T, Extent> subspan(size_type offset, size_type count) const noexcept
  {
    return span<T, Extent>(_data + offset, count);
  }

 private:
  pointer _data;
  size_type _size;
};  // namespace detail
}  // namespace detail
}  // namespace cudf
