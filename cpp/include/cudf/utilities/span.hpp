/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <limits>
#include <type_traits>

namespace cudf {

constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

namespace detail {

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

}  // namespace detail

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

template <typename T, std::size_t Extent = cudf::dynamic_extent>
struct host_span : public cudf::detail::span_base<T, Extent, host_span<T, Extent>> {
  using base = cudf::detail::span_base<T, Extent, host_span<T, Extent>>;
  using base::base;

  constexpr host_span() noexcept : base() {}  // required to compile on centos

  // Constructor from container
  template <
    typename C,
    // Only supported containers of types convertible to T
    std::enable_if_t<is_host_span_supported_container<C>::value &&
                     std::is_convertible<std::remove_pointer_t<decltype(thrust::raw_pointer_cast(
                                           std::declval<C&>().data()))> (*)[],
                                         T (*)[]>::value>* = nullptr>
  constexpr host_span(C& in) : base(in.data(), in.size())
  {
  }

  // Constructor from const container
  template <
    typename C,
    // Only supported containers of types convertible to T
    std::enable_if_t<is_host_span_supported_container<C>::value &&
                     std::is_convertible<std::remove_pointer_t<decltype(thrust::raw_pointer_cast(
                                           std::declval<C&>().data()))> (*)[],
                                         T (*)[]>::value>* = nullptr>
  constexpr host_span(C const& in) : base(in.data(), in.size())
  {
  }

  // Copy construction to support const conversion
  template <typename OtherT,
            std::size_t OtherExtent,
            typename std::enable_if<(Extent == OtherExtent || Extent == dynamic_extent) &&
                                      std::is_convertible<OtherT (*)[], T (*)[]>::value,
                                    void>::type* = nullptr>
  constexpr host_span(const host_span<OtherT, OtherExtent>& other) noexcept
    : base(other.data(), other.size())
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

template <typename T, std::size_t Extent = cudf::dynamic_extent>
struct device_span : public cudf::detail::span_base<T, Extent, device_span<T, Extent>> {
  using base = cudf::detail::span_base<T, Extent, device_span<T, Extent>>;
  using base::base;

  constexpr device_span() noexcept : base() {}  // required to compile on centos

  template <
    typename C,
    // Only supported containers of types convertible to T
    std::enable_if_t<is_device_span_supported_container<C>::value &&
                     std::is_convertible<std::remove_pointer_t<decltype(thrust::raw_pointer_cast(
                                           std::declval<C&>().data()))> (*)[],
                                         T (*)[]>::value>* = nullptr>
  constexpr device_span(C& in) : base(thrust::raw_pointer_cast(in.data()), in.size())
  {
  }

  template <
    typename C,
    // Only supported containers of types convertible to T
    std::enable_if_t<is_device_span_supported_container<C>::value &&
                     std::is_convertible<std::remove_pointer_t<decltype(thrust::raw_pointer_cast(
                                           std::declval<C&>().data()))> (*)[],
                                         T (*)[]>::value>* = nullptr>
  constexpr device_span(C const& in) : base(thrust::raw_pointer_cast(in.data()), in.size())
  {
  }

  template <typename OtherT,
            std::size_t OtherExtent,
            typename std::enable_if<(Extent == OtherExtent || Extent == dynamic_extent) &&
                                      std::is_convertible<OtherT (*)[], T (*)[]>::value,
                                    void>::type* = nullptr>
  constexpr device_span(const device_span<OtherT, OtherExtent>& other) noexcept
    : base(other.data(), other.size())
  {
  }
};

namespace detail {

/**
 * @brief Generic class for row-major 2D spans. Not compliant with STL container semantics/syntax.
 *
 * The index operator returns the corresponding row.
 */
template <typename T, template <typename, std::size_t> typename RowType>
class base_2dspan {
 public:
  using size_type = std::pair<size_t, size_t>;

  constexpr base_2dspan() noexcept = default;
  constexpr base_2dspan(T* data, size_t rows, size_t columns) noexcept
    : _data{data}, _size{rows, columns}
  {
  }
  base_2dspan(T* data, size_type size) noexcept : _data{data}, _size{size} {}

  constexpr auto data() const noexcept { return _data; }
  constexpr auto size() const noexcept { return _size; }
  constexpr auto count() const noexcept { return size().first * size().second; }
  constexpr bool is_empty() const noexcept { return count() == 0; }

  static constexpr size_t flatten_index(size_t row, size_t column, size_type size) noexcept
  {
    return row * size.second + column;
  }

  constexpr RowType<T, dynamic_extent> operator[](size_t row)
  {
    return {this->data() + flatten_index(row, 0, this->size()), this->size().second};
  }

  template <typename OtherT,
            template <typename, size_t>
            typename OtherRowType,
            typename std::enable_if<std::is_convertible<OtherRowType<OtherT, dynamic_extent>,
                                                        RowType<T, dynamic_extent>>::value,
                                    void>::type* = nullptr>
  constexpr base_2dspan(base_2dspan<OtherT, OtherRowType> const& other) noexcept
    : _data{other.data()}, _size{other.size()}
  {
  }

 protected:
  T* _data = nullptr;
  size_type _size{0, 0};
};

/**
 * @brief Alias for the 2D span for host data.
 *
 * Index operator returns rows as `host_span`.
 */
template <class T>
using host_2dspan = base_2dspan<T, host_span>;

/**
 * @brief Alias for the 2D span for device data.
 *
 * Index operator returns rows as `device_span`.
 */
template <class T>
using device_2dspan = base_2dspan<T, device_span>;

}  // namespace detail
}  // namespace cudf
