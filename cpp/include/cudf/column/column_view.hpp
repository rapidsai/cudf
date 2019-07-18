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

#include <cudf/types.hpp>

#include <vector>

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief A non-owning view of device data as a column of elements,
 * some of which may be null as indicated by a bitmask.
 *
 * A `column_view` can be constructed implicitly from a `cudf::column`, or may
 * be constructed explicitly from a pointer to pre-existing device memory.
 *
 * Unless otherwise noted, the memory layout of the `column_view`'s data and
 * bitmask is expected to adhere to the Arrow Physical Memory Layout
 * Specification: https://arrow.apache.org/docs/memory_layout.html
 *
 * Because `column_view` is non-owning, no device memory is allocated nor free'd
 * when `column_view` objects are created or destroyed.
 *
 * To enable zero-copy slicing, a `column_view` has an `offset` that indicates
 * the index of the first element in the column relative to the base device
 * memory allocation. By default, `offset()` is zero.
 *
 *---------------------------------------------------------------------------**/
class column_view {
 public:
  column_view() = default;
  ~column_view() = default;
  column_view(column_view const&) = default;
  column_view(column_view&&) = default;
  column_view& operator=(column_view const&) = default;
  column_view& operator=(column_view&&) = default;

  /**---------------------------------------------------------------------------*
   * @brief Construct a `column_view` from pointers to device memory for the
   * elements and bitmask of the column.
   *
   * @throws `cudf::logic_error` if `size < 0`
   * @throws `cudf::logic_error` if `size > 0` but `data == nullptr`
   * @throws `cudf::logic_error` if `type == EMPTY` but `data != nullptr` or
   * `null_mask != nullptr`
   * @throws `cudf::logic_error` if `null_count > 0`, but `null_mask == nullptr`
   * @throws `cudf::logic_error` if `offset < 0`
   *
   * @param type The element type
   * @param size The number of elements
   * @param data Pointer to device memory containing the column elements
   * @param null_mask Pointer to device memory containing the null indicator
   * bitmask, may be `nullptr` if `null_count == 0`
   * @param null_count The number of null elements
   * @param offset optional, index of the first element 
   * @param children optional, depending on the element type, child columns may
   * contain additional data
   *---------------------------------------------------------------------------**/
  column_view(data_type type, size_type size, void* data,
              bitmask_type* null_mask, size_type null_count,
              size_type offset = 0,
              std::vector<column_view> const& children = {});

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of a
   * column, and instead, accessing the elements should be done via `data<T>()`.
   *
   * @tparam The type to cast to
   * @return T const* Typed pointer to underlying data
   *---------------------------------------------------------------------------**/
  template <typename T = void>
  T const* head() const noexcept {
    return static_cast<T const*>(_data);
  }

  template <typename T = void>
  T* head() noexcept {
    return static_cast<T*>(_data);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @TODO Clarify behavior for variable-width types.
   *
   * @tparam T The type to cast to
   * @return T const* Typed pointer to underlying data, including the offset
   *---------------------------------------------------------------------------**/
  template <typename T>
  T const* data() const noexcept {
    return head<T>() + _offset;
  }

  template <typename T>
  T* data() noexcept {
    return head<T>() + _offset;
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of elements in the column
   *---------------------------------------------------------------------------**/
  size_type size() const noexcept { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the element `data_type`
   *---------------------------------------------------------------------------**/
  data_type type() const noexcept { return _type; }

  /**---------------------------------------------------------------------------*
   * @brief Indicates if the column can contain null elements, i.e., if it has
   * an allocated bitmask.
   *
   * @note If `null_count() > 0`, this function must always return `true`.
   *
   * @return true The bitmask is allocated
   * @return false The bitmask is not allocated
   *---------------------------------------------------------------------------**/
  bool nullable() const noexcept { return nullptr != _null_mask; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the count of null elements
   *---------------------------------------------------------------------------**/
  size_type null_count() const noexcept { return _null_count; }

  /**---------------------------------------------------------------------------*
   * @brief Indicates if the column contains null elements,
   * i.e., `null_count() > 0`
   *
   * @return true The column contains null elements
   * @return false All elements are valid
   *---------------------------------------------------------------------------**/
  bool has_nulls() const noexcept { return _null_count > 0; }

  /**---------------------------------------------------------------------------*
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   *---------------------------------------------------------------------------**/
  bitmask_type const* null_mask() const noexcept { return _null_mask; }

  bitmask_type* null_mask() noexcept { return _null_mask; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the index of the first element relative to the base memory
   * allocation, i.e., what is returned from `head<T>()`.
   *---------------------------------------------------------------------------**/
  size_type offset() const noexcept { return _offset; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return column_view The requested child `column_view`
   *---------------------------------------------------------------------------**/
  column_view const& child(size_type child_index) const noexcept {
    return _children[child_index];
  }

  column_view& child(size_type child_index) noexcept {
    return _children[child_index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of child columns.
   *---------------------------------------------------------------------------**/
  size_type num_children() const noexcept { return _children.size(); }

 private:
  data_type _type{EMPTY};      ///< Element type
  cudf::size_type _size{};     ///< Number of elements
  void* _data{};               ///< Pointer to device memory containing elements
  bitmask_type* _null_mask{};  ///< Pointer to device memory containing
                               ///< bitmask representing null elements.
                               ///< Optional if `null_count() == 0`
  size_type _null_count{};     ///< The number of null elements
  size_type _offset{};         ///< Index position of the first element.
                               ///< Enables zero-copy slicing
  std::vector<column_view> _children{};  ///< Based on element type, children
                                         ///< may contain additional data
};

}  // namespace cudf