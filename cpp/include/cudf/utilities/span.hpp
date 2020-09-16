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
template <typename T, size_t Extent, typename Derived>
class span {
  static_assert(Extent == dynamic_extent, "Only dynamic extent is supported");

 public:
  using element_type    = T;
  using value_type      = std::remove_cv<T>;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer         = T*;
  using iterator        = T*;
  using const_pointer   = T const*;
  using reference       = T&;
  using const_reference = T const&;

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

  constexpr Derived first(size_type count) const noexcept { return Derived(_data, count); }

  constexpr Derived last(size_type count) const noexcept
  {
    return Derived(_data + _size - count, count);
  }

  constexpr Derived subspan(size_type offset, size_type count) const noexcept
  {
    return Derived(_data + offset, count);
  }

 private:
  pointer _data;
  size_type _size;
};

template <typename T, size_t Extent = dynamic_extent>
struct host_span : public span<T, Extent, host_span<T, Extent>> {
  constexpr host_span() noexcept : span<T, Extent, host_span<T, Extent>>() {}
  constexpr host_span(typename span<T, Extent, host_span<T, Extent>>::pointer data,
                      typename span<T, Extent, host_span<T, Extent>>::size_type size)
    : span<T, Extent, host_span<T, Extent>>(data, size)
  {
  }
  // constexpr host_span(pointer begin, pointer end) : _data(begin), _size(end - begin) {}
  constexpr host_span(const host_span& other) noexcept
    : span<T, Extent, host_span<T, Extent>>(other)
  {
  }
};

template <typename T, size_t Extent = dynamic_extent>
struct device_span : public span<T, Extent, host_span<T, Extent>> {
  constexpr device_span() noexcept : span<T, Extent, device_span<T, Extent>>() {}
  constexpr device_span(typename span<T, Extent, device_span<T, Extent>>::pointer data,
                        typename span<T, Extent, device_span<T, Extent>>::size_type size)
    : span<T, Extent, device_span<T, Extent>>(data, size)
  {
  }
  // constexpr device_span(pointer begin, pointer end) : _data(begin), _size(end - begin) {}
  constexpr device_span(const device_span& other) noexcept
    : span<T, Extent, device_span<T, Extent>>(other)
  {
  }
};

template <typename T, typename Derived>
class base {
  base() : _data(nullptr) {}

 private:
  T* _data;
};

template <typename T>
class derived_a : public base<T, derived_a<T>> {
};

template <typename T>
class derived_b : public base<T, derived_b<T>> {
};

}  // namespace detail
}  // namespace cudf
