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

#include <cuda_runtime.h>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/types.hpp>

namespace cudf {

/**
 * @brief A non-owning, immutable view of device data that represents
 * a list of elements of arbitrary type (including further nested lists).
 *
 */
class list_device_view {
  using lists_column_device_view = cudf::detail::lists_column_device_view;

 public:
  list_device_view() = default;

  CUDA_DEVICE_CALLABLE list_device_view(lists_column_device_view const& lists_column,
                                        size_type const& row_index)
    : lists_column(lists_column), _row_index(row_index)
  {
    column_device_view const& offsets = lists_column.offsets();
    release_assert(row_index >= 0 && row_index < lists_column.size() &&
                   row_index < offsets.size() && "row_index out of bounds");

    begin_offset = offsets.element<size_type>(row_index);
    release_assert(begin_offset >= 0 && begin_offset <= lists_column.child().size() &&
                   "begin_offset out of bounds.");
    _size = offsets.element<size_type>(row_index + 1) - begin_offset;
  }

  ~list_device_view() = default;

  /**
   * @brief Fetches the offset in the list column's child that corresponds to
   * the element at the specified list index.
   *
   * Consider the following lists column:
   *  [
   *   [0,1,2],
   *   [3,4,5],
   *   [6,7,8]
   *  ]
   *
   * The list's internals would look like:
   *  offsets: [0, 3, 6, 9]
   *  child  : [0, 1, 2, 3, 4, 5, 6, 7, 8]
   *
   * The second list row (i.e. row_index=1) is [3,4,5].
   * The third element (i.e. idx=2) of the second list row is 5.
   *
   * The offset of this element as stored in the child column (i.e. 5)
   * may be fetched using this method.
   */
  CUDA_DEVICE_CALLABLE size_type element_offset(size_type idx) const
  {
    release_assert(idx >= 0 && idx < size() && "idx out of bounds");
    return begin_offset + idx;
  }

  /**
   * @brief Fetches the element at the specified index within the list row.
   *
   * @tparam The type of the list's element.
   * @param The index into the list row
   * @return The element at the specified index of the list row.
   */
  template <typename T>
  CUDA_DEVICE_CALLABLE T element(size_type idx) const
  {
    return lists_column.child().element<T>(element_offset(idx));
  }

  /**
   * @brief Checks whether element is null at specified index in the list row.
   */
  CUDA_DEVICE_CALLABLE bool is_null(size_type idx) const
  {
    release_assert(idx >= 0 && idx < size() && "Index out of bounds.");
    auto element_offset = begin_offset + idx;
    return lists_column.child().is_null(element_offset);
  }

  /**
   * @brief Checks whether this list row is null.
   */
  CUDA_DEVICE_CALLABLE bool is_null() const { return lists_column.is_null(_row_index); }

  /**
   * @brief Fetches the number of elements in this list row.
   */
  CUDA_DEVICE_CALLABLE size_type size() const { return _size; }

  /**
   * @brief Fetches the lists_column_device_view that contains this list.
   */
  CUDA_DEVICE_CALLABLE lists_column_device_view const& get_column() const { return lists_column; }

 private:
  lists_column_device_view const& lists_column;
  size_type _row_index{};  // Row index in the Lists column vector.
  size_type _size{};       // Number of elements in *this* list row.

  size_type begin_offset;  // Offset in list_column_device_view where this list begins.
};

}  // namespace cudf
