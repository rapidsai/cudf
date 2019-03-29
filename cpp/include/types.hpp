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
#ifndef TYPES_HPP
#define TYPES_HPP

#include <utilities/error_utils.hpp>
#include "cudf.h"
#include <algorithm>
#include <cassert>

namespace cudf {

/**
 * @brief A wrapper for a set of gdf_columns of equal number of rows.
 *
 */
struct table {
  /**---------------------------------------------------------------------------*
   * @brief Constructs a table object from an array of `gdf_column`s
   *
   * @param cols The array of columns wrapped by the table
   * @param num_cols  The number of columns in the array
   *---------------------------------------------------------------------------**/
  table(gdf_column* cols[], gdf_size_type num_cols)
      : columns{cols}, _num_columns{num_cols} {
    CUDF_EXPECTS(nullptr != cols[0], "Null input column");

    this->_num_rows = cols[0]->size;

    std::for_each(columns, columns + _num_columns, [this](gdf_column* col) {
      CUDF_EXPECTS(nullptr != col, "Null input column");
      CUDF_EXPECTS(_num_rows == col->size, "Column size mismatch");
    });
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns const pointer to const of the first `gdf_column` in the
   * table.
   *
   *---------------------------------------------------------------------------**/
  gdf_column const* const* begin() const { return columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the first `gdf_column` in the table.
   *
   *---------------------------------------------------------------------------**/
  gdf_column** begin() { return columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns const pointer to const of one past the last `gdf_column` in
   * the table
   *
   *---------------------------------------------------------------------------**/
  gdf_column const* const* end() const { return columns + _num_columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to one past the last `gdf_column` in the table
   *
   *---------------------------------------------------------------------------**/
  gdf_column** end() { return columns + _num_columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the column specified by an index.
   *
   * @param index The index of the desired column
   * @return gdf_column* Pointer to the column at `index`
   *---------------------------------------------------------------------------**/
  gdf_column* get_column(gdf_index_type index) {
    assert(index < _num_columns);
    return columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer const of the column specified by an index.
   *
   * @param index The index of the desired column
   * @return gdf_column* Pointer to the column at `index`
   *---------------------------------------------------------------------------**/
  gdf_column const* get_column(gdf_index_type index) const {
    return columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of columns in the table
   *
   *---------------------------------------------------------------------------**/
  gdf_size_type num_columns() const { return _num_columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of rows in the table
   *
   *---------------------------------------------------------------------------**/
  gdf_size_type num_rows() const { return _num_rows; }

 private:
  gdf_column** columns;              ///< The set of gdf_columns
  gdf_size_type const _num_columns;  ///< The number of columns in the set
  gdf_size_type _num_rows;     ///< The number of elements in each column
};

/**
 * @brief A wrapper for a set of gdf_columns of any number of rows.
 *
 */
struct column_array {
  /**---------------------------------------------------------------------------*
   * @brief Constructs a column_array object from an array of `gdf_column`s
   *
   * @param cols The array of columns wrapped by the column_array
   * @param num_cols  The number of columns in the array
   *---------------------------------------------------------------------------**/
  column_array(gdf_column* cols[], gdf_size_type num_cols)
      : columns{cols}, _num_columns{num_cols}
  { }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the first `gdf_column` in the column_array.
   *
   *---------------------------------------------------------------------------**/
  gdf_column** begin() { return columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns const pointer to const of one past the last `gdf_column` in
   * the column_array
   *
   *---------------------------------------------------------------------------**/
  gdf_column const* const* end() const { return columns + _num_columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to one past the last `gdf_column` in the column_array
   *
   *---------------------------------------------------------------------------**/
  gdf_column** end() { return columns + _num_columns; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the column specified by an index.
   *
   * @param index The index of the desired column
   * @return gdf_column* Pointer to the column at `index`
   *---------------------------------------------------------------------------**/
  gdf_column* get_column(gdf_index_type index) {
    assert(index < _num_columns);
    return columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer const of the column specified by an index.
   *
   * @param index The index of the desired column
   * @return gdf_column* Pointer to the column at `index`
   *---------------------------------------------------------------------------**/
  gdf_column const* get_column(gdf_index_type index) const {
    return columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of columns in the column_array
   *
   *---------------------------------------------------------------------------**/
  gdf_size_type num_columns() const { return _num_columns; }

 private:
  gdf_column** columns;             /**< The set of gdf_columns*/
  gdf_size_type const _num_columns; /**< The number of columns in the set */
};

}  // namespace cudf

#endif
