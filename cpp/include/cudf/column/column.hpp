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

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include "column_view.hpp"

#include <rmm/device_buffer.hpp>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Class definition for cudf::column
 */

namespace cudf {

/**
 * @brief A container of nullable device data as a column of elements.
 *
 * @ingroup column_classes Column
 * @{
 */

class column {
 public:
  column()        = default;
  ~column()       = default;
  column& operator=(column const& other) = delete;
  column& operator=(column&& other) = delete;

  /**
   * @brief Construct a new column by deep copying the contents of `other`.
   *
   * All device memory allocation and copying is done using the
   * `device_memory_resource` and `stream` from `other`.
   *
   * @param other The column to copy
   **/
  column(column const& other);

  /**
   * @brief Construct a new column object by deep copying the contents of
   *`other`.
   *
   * Uses the specified `stream` and device_memory_resource for all allocations
   * and copies.
   *
   * @param other The `column` to copy
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for all device memory allocations
   */
  column(column const& other,
         cudaStream_t stream,
         rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Move the contents from `other` to create a new column.
   *
   * After the move, `other.size() == 0` and `other.type() = {EMPTY}`
   *
   * @param other The column whose contents will be moved into the new column
   **/
  column(column&& other) noexcept;

  /**
   * @brief Construct a new column from existing device memory.
   *
   * @note This constructor is primarily intended for use in column factory
   * functions.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements in the column
   * @param[in] data The column's data
   * @param[in] null_mask Optional, column's null value indicator bitmask. May
   * be empty if `null_count` is 0 or `UNKNOWN_NULL_COUNT`.
   * @param null_count Optional, the count of null elements. If unknown, specify
   * `UNKNOWN_NULL_COUNT` to indicate that the null count should be computed on
   * the first invocation of `null_count()`.
   * @param children Optional, vector of child columns
   **/
  template <typename B1, typename B2 = rmm::device_buffer>
  column(data_type dtype,
         size_type size,
         B1&& data,
         B2&& null_mask                                  = {},
         size_type null_count                            = UNKNOWN_NULL_COUNT,
         std::vector<std::unique_ptr<column>>&& children = {})
    : _type{dtype},
      _size{size},
      _data{std::forward<B1>(data)},
      _null_mask{std::forward<B2>(null_mask)},
      _null_count{null_count},
      _children{std::move(children)}
  {
  }

  /**
   * @brief Construct a new column by deep copying the contents of a
   * `column_view`.
   *
   * This accounts for the `column_view`'s offset.
   *
   * @param view The view to copy
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for all device memory allocations
   */
  explicit column(column_view view,
                  cudaStream_t stream                 = 0,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Returns the column's logical element type
   */
  data_type type() const noexcept { return _type; }

  /**
   * @brief Returns the number of elements
   */
  size_type size() const noexcept { return _size; }

  /**
   * @brief Returns the count of null elements.
   *
   * @note If the column was constructed with `UNKNOWN_NULL_COUNT`, or if at any
   * point `set_null_count(UNKNOWN_NULL_COUNT)` was invoked, then the
   * first invocation of `null_count()` will compute and store the count of null
   * elements indicated by the `null_mask` (if it exists).
   */
  size_type null_count() const;

  /**
   * @brief Sets the column's null value indicator bitmask to `new_null_mask`.
   *
   * @throws cudf::logic_error if new_null_count is larger than 0 and the size
   * of `new_null_mask` does not match the size of this column.
   *
   * @param new_null_mask New null value indicator bitmask (rvalue overload &
   * moved) to set the column's null value indicator mask. May be empty if
   * `new_null_count` is 0 or `UNKOWN_NULL_COUNT`.
   * @param new_null_count Optional, the count of null elements. If unknown,
   * specify `UNKNOWN_NULL_COUNT` to indicate that the null count should be
   * computed on the first invocation of `null_count()`.
   */
  void set_null_mask(rmm::device_buffer&& new_null_mask,
                     size_type new_null_count = UNKNOWN_NULL_COUNT);

  /**
   * @brief Sets the column's null value indicator bitmask to `new_null_mask`.
   *
   * @throws cudf::logic_error if new_null_count is larger than 0 and the size
   * of `new_null_mask` does not match the size of this column.
   *
   * @param new_null_mask New null value indicator bitmask (lvalue overload &
   * copied) to set the column's null value indicator mask. May be empty if
   * `new_null_count` is 0 or `UNKOWN_NULL_COUNT`.
   * @param new_null_count Optional, the count of null elements. If unknown,
   * specify `UNKNOWN_NULL_COUNT` to indicate that the null count should be
   * computed on the first invocation of `null_count()`.
   */
  void set_null_mask(rmm::device_buffer const& new_null_mask,
                     size_type new_null_count = UNKNOWN_NULL_COUNT);

  /**
   * @brief Updates the count of null elements.
   *
   * @note `UNKNOWN_NULL_COUNT` can be specified as `new_null_count` to force
   * the next invocation of `null_count()` to recompute the null count from the
   * null mask.
   *
   * @throws cudf::logic_error if `new_null_count > 0 and nullable() == false`
   *
   * @param new_null_count The new null count.
   */
  void set_null_count(size_type new_null_count);

  /**
   * @brief Indicates whether it is possible for the column to contain null
   * values, i.e., it has an allocated null mask.
   *
   * This may return `false` iff `null_count() == 0`.
   *
   * May return true even if `null_count() == 0`. This function simply indicates
   * whether the column has an allocated null mask.
   *
   * @return true The column can hold null values
   * @return false The column cannot hold null values
   */
  bool nullable() const noexcept { return (_null_mask.size() > 0); }

  /**
   * @brief Indicates whether the column contains null elements.
   *
   * @return true One or more elements are null
   * @return false Zero elements are null
   */
  bool has_nulls() const noexcept { return (null_count() > 0); }

  /**
   * @brief Returns the number of child columns
   */
  size_type num_children() const noexcept { return _children.size(); }

  /**
   * @brief Returns a reference to the specified child
   *
   * @param child_index Index of the desired child
   * @return column& Reference to the desired child
   */
  column& child(size_type child_index) noexcept { return *_children[child_index]; };

  /**
   * @brief Returns a const reference to the specified child
   *
   * @param child_index Index of the desired child
   * @return column const& Const reference to the desired child
   */
  column const& child(size_type child_index) const noexcept { return *_children[child_index]; };

  /**
   * @brief Wrapper for the contents of a column.
   *
   * Returned by `column::release()`.
   */
  struct contents {
    std::unique_ptr<rmm::device_buffer> data;
    std::unique_ptr<rmm::device_buffer> null_mask;
    std::vector<std::unique_ptr<column>> children;
  };

  /**
   * @brief Releases ownership of the column's contents.
   *
   * It is the caller's responsibility to query the `size(), null_count(),
   * type()` before invoking `release()`.
   *
   * After calling `release()` on a column it will be empty, i.e.:
   * - `type() == data_type{EMPTY}`
   * - `size() == 0`
   * - `null_count() == 0`
   * - `num_children() == 0`
   *
   * @return A `contents` struct containing the data, null mask, and children of
   * the column.
   */
  contents release() noexcept;

  /**
   * @brief Creates an immutable, non-owning view of the column's data and
   * children.
   *
   * @return column_view The immutable, non-owning view
   */
  column_view view() const;

  /**
   * @brief Implicit conversion operator to a `column_view`.
   *
   * This allows passing a `column` object directly into a function that
   * requires a `column_view`. The conversion is automatic.
   *
   * @return column_view Immutable, non-owning `column_view`
   */
  operator column_view() const { return this->view(); };

  /**
   * @brief Creates a mutable, non-owning view of the column's data and
   * children.
   *
   * @note Creating a mutable view of a `column` invalidates the `column`'s
   * `null_count()` by setting it to `UNKNOWN_NULL_COUNT`. The user can
   * either explicitly update the null count with `set_null_count()`, or
   * if not, the null count will be recomputed on the next invocation of
   *`null_count()`.
   *
   * @return mutable_column_view The mutable, non-owning view
   */
  mutable_column_view mutable_view();

  /**
   * @brief Implicit conversion operator to a `mutable_column_view`.
   *
   * This allows pasing a `column` object into a function that accepts a
   *`mutable_column_view`. The conversion is automatic.

   * @note Creating a mutable view of a `column` invalidates the `column`'s
   * `null_count()` by setting it to `UNKNOWN_NULL_COUNT`. For best performance,
   * the user should explicitly update the null count with `set_null_count()`.
   * Otherwise, the null count will be recomputed on the next invocation of
   * `null_count()`.
   *
   * @return mutable_column_view Mutable, non-owning `mutable_column_view`
   */
  operator mutable_column_view() { return this->mutable_view(); };

 private:
  cudf::data_type _type{type_id::EMPTY};  ///< Logical type of elements in the column
  cudf::size_type _size{};                ///< The number of elements in the column
  rmm::device_buffer _data{};             ///< Dense, contiguous, type erased device memory
                                          ///< buffer containing the column elements
  rmm::device_buffer _null_mask{};        ///< Bitmask used to represent null values.
                                          ///< May be empty if `null_count() == 0`
  mutable cudf::size_type _null_count{UNKNOWN_NULL_COUNT};  ///< The number of null elements
  std::vector<std::unique_ptr<column>> _children{};         ///< Depending on element type, child
                                                            ///< columns may contain additional data
};

/** @} */  // end of group
}  // namespace cudf
