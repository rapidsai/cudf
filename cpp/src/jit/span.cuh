/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
#include <cudf/utilities/bit.hpp>

namespace cudf {

namespace jit {

/**
 * @brief C++20 std::span with reduced feature set.
 *
 */
template <typename T>
struct device_span {
  using element_type = T;  ///< The type of the elements in the span

 private:
  element_type* _data = nullptr;
  size_t _size        = 0;

 public:
  CUDF_HOST_DEVICE constexpr device_span() {}

  /**
   * @brief Constructs a span from a pointer and a size.
   *
   * @param data Pointer to the first element in the span.
   * @param size The number of elements in the span.
   */
  CUDF_HOST_DEVICE constexpr device_span(element_type* data, size_t size) : _data{data}, _size{size}
  {
  }

  /**
   * @brief Returns a pointer to the beginning of the sequence.
   *
   * @return A pointer to the first element of the span
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr element_type* data() const { return _data; }

  /**
   * @brief Returns the number of elements in the span.
   *
   * @return The number of elements in the span
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr size_t size() const { return _size; }

  /**
   * @brief Checks if the span is empty.
   *
   * @return True if the span is empty, false otherwise
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr bool empty() const { return _size == 0; }

  /**
   * @brief Returns a reference to the idx-th element of the sequence.
   *
   * The behavior is undefined if idx is out of range (i.e., if it is greater than or equal to
   * size()).
   *
   * @param idx the index of the element to access
   * @return A reference to the idx-th element of the sequence, i.e., `data()[idx]`
   */
  CUDF_HOST_DEVICE constexpr element_type& operator[](size_t idx) const { return _data[idx]; }

  /**
   * @brief Returns an iterator to the first element of the span.
   *
   * If the span is empty, the returned iterator will be equal to end().
   *
   * @return An iterator to the first element of the span
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr element_type* begin() const { return _data; }

  /**
   * @brief Returns an iterator to the element following the last element of the span.
   *
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   *
   * @return An iterator to the element following the last element of the span
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr element_type* end() const { return _data + _size; }

  CUDF_HOST_DEVICE [[nodiscard]] constexpr device_span<T const> as_const() const
  {
    return device_span<T const>{_data, _size};
  }
};

/**
 * @brief A span type with optional/nullable elements.
 *
 * Optional implies the span contains nullable elements.
 * The nullability of the elements is internally represented by an optional bitmask which can be
 * nullptr when all the elements are non-null.
 */
template <typename T>
struct device_optional_span : device_span<T> {
 private:
  using base               = device_span<T>;
  bitmask_type* _null_mask = nullptr;

 public:
  CUDF_HOST_DEVICE constexpr device_optional_span() {}

  /**
   * @brief Constructs an optional span from a span and a null-mask.
   *
   * @param span Span containing the elements
   * @param null_mask The null-mask determining the validity of the elements or nullptr if all
   * valid.
   */
  CUDF_HOST_DEVICE device_optional_span(device_span<T> span, bitmask_type* null_mask)
    : base{span}, _null_mask{null_mask}
  {
  }

  /// @copydoc column_device_view::nullable
  [[nodiscard]] CUDF_HOST_DEVICE bool nullable() const { return _null_mask != nullptr; }

#ifdef __CUDACC__

  /// @copydoc column_device_view::is_valid_nocheck
  [[nodiscard]] __device__ bool is_valid_nocheck(size_t element_index) const
  {
    return bit_is_set(_null_mask, element_index);
  }

  /// @copydoc column_device_view::is_valid
  [[nodiscard]] __device__ bool is_valid(size_t element_index) const
  {
    return not nullable() or is_valid_nocheck(element_index);
  }

  /// @copydoc column_device_view::is_null
  [[nodiscard]] __device__ bool is_null(size_t element_index) const
  {
    return !is_valid(element_index);
  }

  /// @brief converts the optional span to a regular non-nullable span.
  [[nodiscard]] __device__ base to_span() const noexcept { return static_cast<base const&>(*this); }

#endif
};

}  // namespace jit
}  // namespace cudf
