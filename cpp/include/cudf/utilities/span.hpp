/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cstddef>
#include <limits>
#include <type_traits>

namespace cudf {
namespace detail {

constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

/**
 * @brief C++20 std::span with reduced feature set.
 */
template <typename T, std::size_t Extent, typename Derived>
class span_base {
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

  static constexpr std::size_t extent = Extent;

  constexpr span_base() noexcept : _data(nullptr), _size(0) {}
  constexpr span_base(pointer data, size_type size) : _data(data), _size(size) {}
  // constexpr span_base(pointer begin, pointer end) : _data(begin), _size(end - begin) {}
  constexpr span_base(span_base const& other) noexcept = default;
  constexpr span_base& operator=(span_base const& other) noexcept = default;

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

  /**
   * @brief Obtains a subspan consisting of the first N elements of the sequence
   *
   * @param count Number of elements from the beginning of this span to put in the subspan.
   */
  constexpr Derived first(size_type count) const noexcept { return Derived(_data, count); }

  /**
   * @brief Obtains a subspan consisting of the last N elements of the sequence
   *
   * @param count Number of elements from the end of this span to put in the subspan
   */
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

// ===== host_span =================================================================================

template <typename T>
struct is_host_span_supported_container : std::false_type {
};

template <typename T, typename Alloc>
struct is_host_span_supported_container<  //
  std::vector<T, Alloc>> : std::true_type {
};

template <typename T, typename Alloc>
struct is_host_span_supported_container<  //
  thrust::host_vector<T, Alloc>> : std::true_type {
};

template <typename T, std::size_t Extent = dynamic_extent>
struct host_span : public span_base<T, Extent, host_span<T, Extent>> {
  using base = cudf::detail::span_base<T, Extent, host_span<T, Extent>>;
  using base::base;

  constexpr host_span() noexcept : base() {}  // required to compile on centos

  template <typename C, std::enable_if_t<is_host_span_supported_container<C>::value>* = nullptr>
  constexpr host_span(C& in) : base(in.data(), in.size())
  {
  }

  template <typename C, std::enable_if_t<is_host_span_supported_container<C>::value>* = nullptr>
  constexpr host_span(C const& in) : base(in.data(), in.size())
  {
  }
};

// ===== device_span ===============================================================================

template <typename T>
struct is_device_span_supported_container : std::false_type {
};

template <typename T, typename Alloc>
struct is_device_span_supported_container<  //
  thrust::device_vector<T, Alloc>> : std::true_type {
};

template <typename T>
struct is_device_span_supported_container<  //
  rmm::device_vector<T>> : std::true_type {
};

template <typename T>
struct is_device_span_supported_container<  //
  rmm::device_uvector<T>> : std::true_type {
};

template <typename T, std::size_t Extent = dynamic_extent>
struct device_span : public span_base<T, Extent, device_span<T, Extent>> {
  using base = cudf::detail::span_base<T, Extent, device_span<T, Extent>>;
  using base::base;

  constexpr device_span() noexcept : base() {}  // required to compile on centos

  template <typename C, std::enable_if_t<is_device_span_supported_container<C>::value>* = nullptr>
  constexpr device_span(C& in) : base(thrust::raw_pointer_cast(in.data()), in.size())
  {
  }

  template <typename C, std::enable_if_t<is_device_span_supported_container<C>::value>* = nullptr>
  constexpr device_span(C const& in) : base(thrust::raw_pointer_cast(in.data()), in.size())
  {
  }
};

}  // namespace detail
}  // namespace cudf
