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

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace experimental {

class table {
 public:
  table() = delete;
  ~table() = default;
  table(table&&) = default;
  table& operator=(table const&) = delete;
  table& operator=(table&&) = delete;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new table by copying the contents of another table.
   *---------------------------------------------------------------------------**/
  table(table const& other);

  /**---------------------------------------------------------------------------*
   * @brief Moves the contents from a vector of `unique_ptr`s to columns to
   * construct a new table.
   *
   * @param columns The vector of `unique_ptr`s to columns whose contents will
   * be moved into the new table.
   *---------------------------------------------------------------------------**/
  table(std::vector<std::unique_ptr<column>>&& columns);

  /**---------------------------------------------------------------------------*
   * @brief Copy the contents of a `table_view` to construct a new `table`.
   *
   * @param view The view whose contents will be copied to create a new `table`
   * @param stream Optional, stream on which all memory allocations and copies
   * will be performed
   * @param mr Optional, the memory resource that will be used for allocating
   * the device memory for the new columns
   *---------------------------------------------------------------------------**/
  table(table_view view, cudaStream_t stream = 0,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of columns in the table
   *---------------------------------------------------------------------------**/
  size_type num_columns() const noexcept { return _columns.size(); }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of rows
   *---------------------------------------------------------------------------**/
  size_type num_rows() const noexcept { return _num_rows; }

  /**---------------------------------------------------------------------------*
   * @brief Returns an immutable, non-owning `table_view` of the contents of
   *this `table`.
   *---------------------------------------------------------------------------**/
  table_view view() const;

  /**---------------------------------------------------------------------------*
   * @brief Conversion operator to an immutable, non-owning `table_view` of the
   * contents of this `table`.
   *---------------------------------------------------------------------------**/
  operator table_view() const { return this->view(); };

  /**---------------------------------------------------------------------------*
   * @brief Returns a mutable, non-owning `mutable_table_view` of the contents
   * of this `table`.
   *---------------------------------------------------------------------------**/
  mutable_table_view mutable_view();

  /**---------------------------------------------------------------------------*
   * @brief Conversion operator to a mutable, non-owning `mutable_table_view` of
   *the contents of this `table`.
   *---------------------------------------------------------------------------**/
  operator mutable_table_view() { return this->mutable_view(); };

  /**---------------------------------------------------------------------------*
   * @brief Releases ownership of the `column`s by returning a vector of
   * `unique_ptr`s to the constituent columns.
   *
   * After `release()`, `num_columns() == 0` and `num_rows() == 0`
   *---------------------------------------------------------------------------**/
  std::vector<std::unique_ptr<column>> release();

  /**---------------------------------------------------------------------------*
   * @brief Returns a table_view with set of specified columns.
   *
   * @throws cudf::logic_error
   * If any element in `column_indices` is outside [0, num_columns())
   *
   * @param column_indices Indices of columns in the table
   * @return A table_view consisting of columns from the original table
   * specified by the elements of `column_indices`
   *---------------------------------------------------------------------------**/
  table_view select(std::vector<cudf::size_type> const& column_indices) const;

 private:
  std::vector<std::unique_ptr<column>> _columns{};
  size_type _num_rows{};
};

/**---------------------------------------------------------------------------*
 * @brief Elements of `tables_to_concat` are concatenated to return single
 * table_view
 *
 * @throws cudf::logic_error
 * If number of rows mismatch
 *
 * @param tables_to_concat The tables to be concatenated into a single
 * table_view
 * @return A single table having all the columns from the elements of
 * `tables_to_concat` respectively in the same order.
 *---------------------------------------------------------------------------**/
table_view concat(std::vector<table_view> const& tables_to_concat);

}  // namespace experimental
}  // namespace cudf
