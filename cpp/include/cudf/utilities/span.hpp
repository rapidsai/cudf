/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup utility_span
 * @{
 * @file
 * @brief APIs for spans
 */

/// A constant used to differentiate std::span of static and dynamic extent
constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

/** @} */  // end of group
namespace detail {

/**
 * @brief C++20 std::span with reduced feature set.
 *
 */
template <typename T, std::size_t Extent, typename Derived>
class span_base {
  static_assert(Extent == dynamic_extent, "Only dynamic extent is supported");

 public:
  using element_type    = T;                  ///< The type of the elements in the span
  using value_type      = std::remove_cv<T>;  ///< Stored value type
  using size_type       = std::size_t;        ///< The type used for the size of the span
  using difference_type = std::ptrdiff_t;     ///< std::ptrdiff_t
  using pointer         = T*;                 ///< The type of the pointer returned by data()
  using iterator        = T*;                 ///< The type of the iterator returned by begin()
  using const_pointer   = T const*;           ///< The type of the pointer returned by data() const
  using reference       = T&;  ///< The type of the reference returned by operator[](size_type)
  using const_reference =
    T const&;  ///< The type of the reference returned by operator[](size_type) const

  static constexpr std::size_t extent = Extent;  ///< The extent of the span

  constexpr span_base() noexcept {}
  /**
   * @brief Constructs a span from a pointer and a size.
   *
   * @param data Pointer to the first element in the span.
   * @param size The number of elements in the span.
   */
  constexpr span_base(pointer data, size_type size) : _data(data), _size(size) {}
  // constexpr span_base(pointer begin, pointer end) : _data(begin), _size(end - begin) {}
  constexpr span_base(span_base const&) noexcept = default;  ///< Copy constructor
  /**
   * @brief Copy assignment operator.
   *
   * @return Reference to this span.
   */
  constexpr span_base& operator=(span_base const&) noexcept = default;

  // not noexcept due to undefined behavior when size = 0
  /**
   * @brief Returns a reference to the first element in the span.
   *
   * Calling front on an empty span results in undefined behavior.
   *
   * @return Reference to the first element in the span
   */
  [[nodiscard]] constexpr reference front() const { return _data[0]; }
  // not noexcept due to undefined behavior when size = 0
  /**
   * @brief Returns a reference to the last element in the span.
   *
   * Calling last on an empty span results in undefined behavior.
   *
   * @return Reference to the last element in the span
   */
  [[nodiscard]] constexpr reference back() const { return _data[_size - 1]; }
  // not noexcept due to undefined behavior when idx < 0 || idx >= size
  /**
   * @brief Returns a reference to the idx-th element of the sequence.
   *
   * The behavior is undefined if idx is out of range (i.e., if it is greater than or equal to
   * size()).
   *
   * @param idx the index of the element to access
   * @return A reference to the idx-th element of the sequence, i.e., `data()[idx]`
   */
  constexpr reference operator[](size_type idx) const { return _data[idx]; }

  /**
   * @brief Returns an iterator to the first element of the span.
   *
   * If the span is empty, the returned iterator will be equal to end().
   *
   * @return An iterator to the first element of the span
   */
  [[nodiscard]] constexpr iterator begin() const noexcept { return _data; }
  /**
   * @brief Returns an iterator to the element following the last element of the span.
   *
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   *
   * @return An iterator to the element following the last element of the span
   */
  [[nodiscard]] constexpr iterator end() const noexcept { return _data + _size; }
  /**
   * @brief Returns a pointer to the beginning of the sequence.
   *
   * @return A pointer to the first element of the span
   */
  [[nodiscard]] constexpr pointer data() const noexcept { return _data; }

  /**
   * @brief Returns the number of elements in the span.
   *
   * @return The number of elements in the span
   */
  [[nodiscard]] constexpr size_type size() const noexcept { return _size; }
  /**
   * @brief Returns the size of the sequence in bytes.
   *
   * @return The size of the sequence in bytes
   */
  [[nodiscard]] constexpr size_type size_bytes() const noexcept { return sizeof(T) * _size; }
  /**
   * @brief Checks if the span is empty.
   *
   * @return True if the span is empty, false otherwise
   */
  [[nodiscard]] constexpr bool empty() const noexcept { return _size == 0; }

  /**
   * @brief Obtains a subspan consisting of the first N elements of the sequence
   *
   * @param count Number of elements from the beginning of this span to put in the subspan.
   * @return A subspan of the first N elements of the sequence
   */
  [[nodiscard]] constexpr Derived first(size_type count) const noexcept
  {
    return Derived(_data, count);
  }

  /**
   * @brief Obtains a subspan consisting of the last N elements of the sequence
   *
   * @param count Number of elements from the end of this span to put in the subspan
   * @return A subspan of the last N elements of the sequence
   */
  [[nodiscard]] constexpr Derived last(size_type count) const noexcept
  {
    return Derived(_data + _size - count, count);
  }

 private:
  pointer _data{nullptr};
  size_type _size{0};
};

}  // namespace detail

/**
 * @addtogroup utility_span
 * @{
 * @file
 * @brief APIs for spans
 */

// ===== host_span =================================================================================

template <typename T>
struct is_host_span_supported_container : std::false_type {};

template <typename T, typename Alloc>
struct is_host_span_supported_container<  //
  std::vector<T, Alloc>> : std::true_type {};

template <typename T, typename Alloc>
struct is_host_span_supported_container<  //
  thrust::host_vector<T, Alloc>> : std::true_type {};

template <typename T, typename Alloc>
struct is_host_span_supported_container<  //
  std::basic_string<T, std::char_traits<T>, Alloc>> : std::true_type {};

/**
 * @brief C++20 std::span with reduced feature set.
 *
 */
template <typename T, std::size_t Extent = cudf::dynamic_extent>
struct host_span : public cudf::detail::span_base<T, Extent, host_span<T, Extent>> {
  using base = cudf::detail::span_base<T, Extent, host_span<T, Extent>>;  ///< Base type
  using base::base;

  constexpr host_span() noexcept : base() {}  // required to compile on centos

  /// Constructor from pointer and size
  /// @param data Pointer to the first element in the span
  /// @param size The number of elements in the span
  /// @param is_device_accessible Whether the data is device accessible (e.g. pinned memory)
  constexpr host_span(T* data, std::size_t size, bool is_device_accessible)
    : base(data, size), _is_device_accessible{is_device_accessible}
  {
  }

  /// Constructor from container
  /// @param in The container to construct the span from
  template <typename C,
            // Only supported containers of types convertible to T
            std::enable_if_t<is_host_span_supported_container<C>::value &&
                             std::is_convertible_v<
                               std::remove_pointer_t<decltype(thrust::raw_pointer_cast(  // NOLINT
                                 std::declval<C&>().data()))> (*)[],
                               T (*)[]>>* = nullptr>  // NOLINT
  constexpr host_span(C& in) : base(thrust::raw_pointer_cast(in.data()), in.size())
  {
  }

  /// Constructor from const container
  /// @param in The container to construct the span from
  template <typename C,
            // Only supported containers of types convertible to T
            std::enable_if_t<is_host_span_supported_container<C>::value &&
                             std::is_convertible_v<
                               std::remove_pointer_t<decltype(thrust::raw_pointer_cast(  // NOLINT
                                 std::declval<C&>().data()))> (*)[],
                               T (*)[]>>* = nullptr>  // NOLINT
  constexpr host_span(C const& in) : base(thrust::raw_pointer_cast(in.data()), in.size())
  {
  }

  /// Constructor from a host_vector
  /// @param in The host_vector to construct the span from
  template <typename OtherT,
            // Only supported containers of types convertible to T
            std::enable_if_t<std::is_convertible_v<OtherT (*)[], T (*)[]>>* = nullptr>  // NOLINT
  constexpr host_span(cudf::detail::host_vector<OtherT>& in)
    : base(in.data(), in.size()), _is_device_accessible{in.get_allocator().is_device_accessible()}
  {
  }

  /// Constructor from a const host_vector
  /// @param in The host_vector to construct the span from
  template <typename OtherT,
            // Only supported containers of types convertible to T
            std::enable_if_t<std::is_convertible_v<OtherT (*)[], T (*)[]>>* = nullptr>  // NOLINT
  constexpr host_span(cudf::detail::host_vector<OtherT> const& in)
    : base(in.data(), in.size()), _is_device_accessible{in.get_allocator().is_device_accessible()}
  {
  }

  // Copy construction to support const conversion
  /// @param other The span to copy
  template <typename OtherT,
            std::size_t OtherExtent,
            std::enable_if_t<(Extent == OtherExtent || Extent == dynamic_extent) &&
                               std::is_convertible_v<OtherT (*)[], T (*)[]>,  // NOLINT
                             void>* = nullptr>
  constexpr host_span(host_span<OtherT, OtherExtent> const& other) noexcept
    : base(other.data(), other.size()), _is_device_accessible{other.is_device_accessible()}
  {
  }

  /**
   * @brief Returns whether the data is device accessible (e.g. pinned memory)
   *
   * @return true if the data is device accessible
   */
  [[nodiscard]] bool is_device_accessible() const { return _is_device_accessible; }

  /**
   * @brief Obtains a span that is a view over the `count` elements of this span starting at offset
   *
   * @param offset The offset of the first element in the subspan
   * @param count The number of elements in the subspan
   * @return A subspan of the sequence, of requested count and offset
   */
  [[nodiscard]] constexpr host_span subspan(typename base::size_type offset,
                                            typename base::size_type count) const noexcept
  {
    return host_span{this->data() + offset, count, _is_device_accessible};
  }

 private:
  bool _is_device_accessible{false};
};

// ===== device_span ===============================================================================

template <typename T>
struct is_device_span_supported_container : std::false_type {};

template <typename T, typename Alloc>
struct is_device_span_supported_container<  //
  thrust::device_vector<T, Alloc>> : std::true_type {};

template <typename T>
struct is_device_span_supported_container<  //
  rmm::device_vector<T>> : std::true_type {};

template <typename T>
struct is_device_span_supported_container<  //
  rmm::device_uvector<T>> : std::true_type {};

/**
 * @brief Device version of C++20 std::span with reduced feature set.
 *
 */
template <typename T, std::size_t Extent = cudf::dynamic_extent>
struct device_span : public cudf::detail::span_base<T, Extent, device_span<T, Extent>> {
  using base = cudf::detail::span_base<T, Extent, device_span<T, Extent>>;  ///< Base type
  using base::base;

  constexpr device_span() noexcept : base() {}  // required to compile on centos

  /// Constructor from container
  /// @param in The container to construct the span from
  template <typename C,
            // Only supported containers of types convertible to T
            std::enable_if_t<is_device_span_supported_container<C>::value &&
                             std::is_convertible_v<
                               std::remove_pointer_t<decltype(thrust::raw_pointer_cast(  // NOLINT
                                 std::declval<C&>().data()))> (*)[],
                               T (*)[]>>* = nullptr>  // NOLINT
  constexpr device_span(C& in) : base(thrust::raw_pointer_cast(in.data()), in.size())
  {
  }

  /// Constructor from const container
  /// @param in The container to construct the span from
  template <typename C,
            // Only supported containers of types convertible to T
            std::enable_if_t<is_device_span_supported_container<C>::value &&
                             std::is_convertible_v<
                               std::remove_pointer_t<decltype(thrust::raw_pointer_cast(  // NOLINT
                                 std::declval<C&>().data()))> (*)[],
                               T (*)[]>>* = nullptr>  // NOLINT
  constexpr device_span(C const& in) : base(thrust::raw_pointer_cast(in.data()), in.size())
  {
  }

  // Copy construction to support const conversion
  /// @param other The span to copy
  template <typename OtherT,
            std::size_t OtherExtent,
            std::enable_if_t<(Extent == OtherExtent || Extent == dynamic_extent) &&
                               std::is_convertible_v<OtherT (*)[], T (*)[]>,  // NOLINT
                             void>* = nullptr>
  constexpr device_span(device_span<OtherT, OtherExtent> const& other) noexcept
    : base(other.data(), other.size())
  {
  }

  /**
   * @brief Obtains a span that is a view over the `count` elements of this span starting at offset
   *
   * @param offset The offset of the first element in the subspan
   * @param count The number of elements in the subspan
   * @return A subspan of the sequence, of requested count and offset
   */
  [[nodiscard]] constexpr device_span subspan(typename base::size_type offset,
                                              typename base::size_type count) const noexcept
  {
    return device_span{this->data() + offset, count};
  }
};
/** @} */  // end of group

namespace detail {

/**
 * @brief Generic class for row-major 2D spans. Not compliant with STL container semantics/syntax.
 *
 * The index operator returns the corresponding row.
 */
template <typename T, template <typename, std::size_t> typename RowType>
class base_2dspan {
 public:
  using size_type =
    std::pair<size_t, size_t>;  ///< Type used to represent the dimension of the span

  constexpr base_2dspan() noexcept = default;
  /**
   * @brief Constructor from a span and number of elements in each row.
   *
   * @param flat_view The flattened 2D span
   * @param columns Number of columns
   */
  constexpr base_2dspan(RowType<T, dynamic_extent> flat_view, size_t columns)
    : _flat{flat_view}, _size{columns == 0 ? 0 : flat_view.size() / columns, columns}
  {
#ifndef __CUDA_ARCH__
    CUDF_EXPECTS(_size.first * _size.second == flat_view.size(), "Invalid 2D span size");
#endif
  }

  /**
   * @brief Returns a pointer to the beginning of the sequence.
   *
   * @return A pointer to the first element of the span
   */
  [[nodiscard]] constexpr auto data() const noexcept { return _flat.data(); }

  /**
   * @brief Returns the size in the span as pair.
   *
   * @return pair representing rows and columns size of the span
   */
  [[nodiscard]] constexpr auto size() const noexcept { return _size; }

  /**
   * @brief Returns the number of elements in the span.
   *
   * @return Number of elements in the span
   */
  [[nodiscard]] constexpr auto count() const noexcept { return _flat.size(); }

  /**
   * @brief Checks if the span is empty.
   *
   * @return True if the span is empty, false otherwise
   */
  [[nodiscard]] constexpr bool is_empty() const noexcept { return count() == 0; }

  /**
   * @brief Returns a reference to the row-th element of the sequence.
   *
   * The behavior is undefined if row is out of range (i.e., if it is greater than or equal to
   * size()).
   *
   * @param row the index of the element to access
   * @return A reference to the row-th element of the sequence, i.e., `data()[row]`
   */
  constexpr RowType<T, dynamic_extent> operator[](size_t row) const
  {
    return _flat.subspan(row * _size.second, _size.second);
  }

  /**
   * @brief Returns a flattened span of the 2D span.
   *
   * @return A flattened span of the 2D span
   */
  [[nodiscard]] constexpr RowType<T, dynamic_extent> flat_view() const { return _flat; }

  /**
   * @brief Construct a 2D span from another 2D span of convertible type
   *
   * @tparam OtherT Type of the other 2D span
   * @tparam OtherRowType Type of the row of the other 2D span
   * @param other The other 2D span
   */
  template <typename OtherT,
            template <typename, size_t>
            typename OtherRowType,
            std::enable_if_t<std::is_convertible_v<OtherRowType<OtherT, dynamic_extent>,
                                                   RowType<T, dynamic_extent>>,
                             void>* = nullptr>
  constexpr base_2dspan(base_2dspan<OtherT, OtherRowType> const& other) noexcept
    : _flat{other.flat_view()}, _size{other.size()}
  {
  }

 protected:
  RowType<T, dynamic_extent> _flat;  ///< flattened 2D span
  size_type _size{0, 0};             ///< num rows, num columns
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
}  // namespace CUDF_EXPORT cudf
