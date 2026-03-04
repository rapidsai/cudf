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

namespace CUDF_EXPORT cudf {

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
};

}  // namespace jit
}  // namespace CUDF_EXPORT cudf
