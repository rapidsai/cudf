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
#pragma once

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include "column_view.hpp"

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <vector>

namespace cudf {

class column {
 public:
  ~column() = default;
  column& operator=(column const& other) = delete;
  column& operator=(column&& other) = delete;
  column() = default;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column by deep copying the device memory of another
   * column.
   *
   * All device memory allocation and copying is done using the
   * `device_memory_resource` and `stream` from `other`.
   *
   * @param other The other column to copy
   *---------------------------------------------------------------------------**/
  column(column const& other) = default;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column object from the contents of another `column`
   * using move semantics. After the move, `other` is guaranteed to be empty.
   *
   * @param other The other column whose device memory will be moved to the new
   * column
   *---------------------------------------------------------------------------**/
  column(column&& other);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column and allocate sufficient storage to hold
   * `size` elements of the specified `type` with an optional null mask
   * allocation.
   *
   * @note This constructor only supports fixed-width, simple types.
   *
   * @throws cudf::logic_error if `type` is not a fixed-width, simple type.
   *
   * @param[in] type The element type
   * @param[in] size The number of elements in the column
   * @param[in] state Optional, controls allocation/initialization of the
   * column's null mask. By default, no null mask is allocated.
   * @param[in] stream Optional stream on which all memory allocation and device
   * kernels will be issued.
   * @param[in] mr Optional resource that will be used for device memory
   * allocation of the column's `data` and `null_mask`.
   *---------------------------------------------------------------------------**/
  column(data_type type, size_type size, mask_state state = UNALLOCATED,
         cudaStream_t stream = 0,
         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column by copying from `device_buffer`s for the
   * column's `data` and `null_mask`.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements in the column
   * @param[in] data `device_buffer` whose contents will be copied for the
   * column's data
   * @param[in] null_mask Optional, `device_buffer` whose contents will be
   * copied for the column's null mask. Buffer may be empty if `null_count` is 0
   * or `UNKNOWN_NULL_COUNT`.
   * @param null_count Optional, the count of null elements. If unknown, specify
   * `UNKNOWN_NULL_COUNT` to indicate that the null count should be computed on
   * the first invocation of `null_count()`.
   * @param children Optional, vector of child columns
   * @param stream Optional, stream on which all memory allocation and copy will
   * be issued.
   * @param mr Optional, `device_memory_resource` that is used for device memory
   * allocation
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer data,
         rmm::device_buffer null_mask = {},
         size_type null_count = UNKNOWN_NULL_COUNT,
         std::vector<column> const& children = {}, cudaStream_t stream = 0,
         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and device_buffers for data and
   * bitmask that will be *shallow* copied.
   *
   * This constructor uses move semantics to take ownership of the
   *device_buffer's device memory. The `device_buffer` passed into this
   *constructor will not longer be valid to use. Furthermore, it will result in
   *undefined behavior if the device_buffer`s associated memory is modified or
   *freed after invoking this constructor.
   *
   * @param dtype The element type
   * @param[in] size The number of elements in the column
   * @param data device_buffer whose data will be moved from into this column
   * @param mask bitmask whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer&& data,
         rmm::device_buffer&& mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, size, and deep copied device
   * buffer for data, and moved bitmask.
   *
   * @param dtype The element type
   * @param size The number of elements
   * @param data device_buffer whose data will be *deep* copied
   * @param mask bitmask whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer data,
         rmm::device_buffer&& mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, size, and moved device
   * buffer for data, and deep copied bitmask.
   *
   * @param dtype The element type
   * @param size The number of elements
   * @param data device_buffer whose data will be moved into this column
   * @param mask bitmask whose data will be deep copied into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer&& data,
         rmm::device_buffer mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column by deep copying from a `column_view`.
   *
   * This accounts for the `column_view`'s offset.
   *
   * @param view The `column_view` that will be copied
   *---------------------------------------------------------------------------**/
  explicit column(
      column_view view, cudaStream_t stream = 0,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Returns the element type
   *---------------------------------------------------------------------------**/
  data_type type() const noexcept { return _type; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of elements
   *---------------------------------------------------------------------------**/
  size_type size() const noexcept { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the count of null elements.
   *
   * @note If the column was constructed with `UNKNOWN_NULL_COUNT`, or if at any
   * point `set_null_count(UNKNOWN_NULL_COUNT)` was invoked, then the
   * first invocation of `null_count()` will compute and store the count of null
   * elements indicated by the `null_mask` (if it exists).
   *---------------------------------------------------------------------------**/
  size_type null_count() const;

  /**---------------------------------------------------------------------------*
   * @brief Updates the count of null elements.
   *
   * @note `UNKNOWN_NULL_COUNT` can be specified as `new_null_count` to force
   * the next invocation of `null_count()` to recompute the null count from the
   * null mask.
   *
   * @throws cudf::logic_error if `new_null_count > 0` but `nullable() == false`
   *
   * @param new_null_count The new null count.
   *---------------------------------------------------------------------------**/
  void set_null_count(size_type new_null_count);

  /**---------------------------------------------------------------------------*
   * @brief Indicates if it is possible for the column to contain null values,
   * i.e., it has an allocated null mask.
   *
   * This may return `false` iff `null_count() == 0`.
   *
   * May return true if `null_count() == 0`. Indicates that the column has an
   * allocated null mask, but all elements are valid.
   *
   * @return true The column can hold null values
   * @return false The column cannot hold null values
   *---------------------------------------------------------------------------**/
  bool nullable() const noexcept { return (_null_mask.size() > 0); }

  /**---------------------------------------------------------------------------*
   * @brief Indicates if the column contains null elements.
   *
   * @return true One or more elements are null
   * @return false All elements are valid
   *---------------------------------------------------------------------------**/
  bool has_nulls() const noexcept { return (null_count() > 0); }

  /**---------------------------------------------------------------------------*
   * @brief Creates an immutable, non-owning view of the column's data and
   * children.
   *
   * @return column_view The immutable, non-owning view
   *---------------------------------------------------------------------------**/
  column_view view() const;

  /**---------------------------------------------------------------------------*
   * @brief Implicit conversion operator to a `column_view`.
   *
   * This allows passing a `column` object directly into a function that
   * requires a `column_view` and the conversion will happen automatically.
   *
   * @return column_view Immutable, non-owning `column_view`
   *---------------------------------------------------------------------------**/
  operator column_view() const { return this->view(); };

  /**---------------------------------------------------------------------------*
   * @brief Creates a mutable, non-owning view of the column's data and
   * children.
   *
   * @note Creating a mutable view of a `column` will invalidate the `column`'s
   * `null_count()` by setting it to `UKNOWN_NULL_COUNT`. This will require the
   * user to either explicitly update the null count with `set_null_count()`,
   * else, the null count to be recomputed on the next invocation of
   *`null_count()`.
   *
   * @return mutable_column_view The mutable, non-owning view
   *---------------------------------------------------------------------------**/
  mutable_column_view mutable_view();

  /**---------------------------------------------------------------------------*
   * @brief Implicit conversion operator to a `mutable_column_view`.
   *
   * This allows pasing a `column` object into a function that accepts a
   *`mutable_column_view` and the conversion will happen automatically.

   * @note Creating a mutable view of a `column` will invalidate the `column`'s
   * `null_count()` by setting it to `UKNOWN_NULL_COUNT`. This will require the
   * user to either explicitly update the null count with `set_null_count()`,
   * else, the null count to be recomputed on the next invocation of
   *`null_count()`.
   *
   * @return mutable_column_view Mutable, non-owning `mutable_column_view`
   *---------------------------------------------------------------------------**/
  operator mutable_column_view() { return this->mutable_view(); };

 private:
  data_type _type{EMPTY};      ///< Logical type of elements in the column
  cudf::size_type _size{};     ///< The number of elements in the column
  rmm::device_buffer _data{};  ///< Dense, contiguous, type erased device memory
                               ///< buffer containing the column elements
  rmm::device_buffer _null_mask{};  ///< Bitmask used to represent null values.
                                    ///< May be empty if `null_count() == 0`
  size_type _null_count{UNKNOWN_NULL_COUNT};  ///< The number of null elements
  std::vector<column> _children{};  ///< Depending on element type, child
                                    ///< columns may contain additional data
};
}  // namespace cudf