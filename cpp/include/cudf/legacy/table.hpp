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

#ifndef TABLE_HPP
#define TABLE_HPP

#include <cudf/types.h>
#include <cudf/types.hpp>

#include <cassert>
#include <initializer_list>
#include <vector>

// Forward declaration
using cudaStream_t = struct CUstream_st*;

namespace cudf {

/**
 * @brief A wrapper for a set of gdf_columns of equal number of rows.
 *
 */
struct table {
  /**---------------------------------------------------------------------------*
   * @brief Construct a table from a vector of `gdf_column` pointers
   *
   * @param cols The vector of columns wrapped by the table
   *---------------------------------------------------------------------------**/
  table(std::vector<gdf_column*> const& cols);

  /**---------------------------------------------------------------------------*
   * @brief Constructs a table object from an array of `gdf_column`s
   *
   * @param cols The array of columns wrapped by the table
   * @param num_cols  The number of columns in the array
   *---------------------------------------------------------------------------**/
  table(gdf_column* cols[], cudf::size_type num_cols);

  /**---------------------------------------------------------------------------*
   * @brief Construct a table from an initializer_list of `gdf_column` pointers
   *
   * @param list The initializer_list of `gdf_column` pointers
   * ---------------------------------------------------------------------------**/
  table(std::initializer_list<gdf_column*> list);

  /**---------------------------------------------------------------------------*
   * @brief Allocates and constructs a set of `gdf_column`s.
   *
   * Allocates an array of `gdf_column`s of the specified size and type.
   *
   * @note It is the caller's responsibility to free the array of gdf_columns
   *and their associated device memory.
   *
   * @param[in] num_rows The size of each gdf_column
   * @param[in] dtypes The type of each column
   * @param[in] dtype_infos The gdf_extra_dtype_info for each column
   * @param[in] allocate_bitmasks If `true`, each column will be allocated an
   * appropriately sized bitmask
   *---------------------------------------------------------------------------**/
  table(cudf::size_type num_rows,
        std::vector<gdf_dtype> const& dtypes,
        std::vector<gdf_dtype_extra_info> const& dtype_infos,
        bool allocate_bitmasks = false,
        bool all_valid         = false,
        cudaStream_t stream    = 0);

  table() = default;

  /**---------------------------------------------------------------------------*
   * @brief Returns const pointer to const of the first `gdf_column` in the
   * table.
   *
   *---------------------------------------------------------------------------**/
  gdf_column const* const* begin() const { return &(*_columns.begin()); }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the first `gdf_column` in the table.
   *
   *---------------------------------------------------------------------------**/
  gdf_column** begin() { return &(*_columns.begin()); }

  /**---------------------------------------------------------------------------*
   * @brief Returns const pointer to const of one past the last `gdf_column` in
   * the table
   *
   *---------------------------------------------------------------------------**/
  gdf_column const* const* end() const { return &(*_columns.end()); }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to one past the last `gdf_column` in the table
   *
   *---------------------------------------------------------------------------**/
  gdf_column** end() { return &(*_columns.end()); }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the column specified by an index.
   *
   * @param index The index of the desired column
   * @return gdf_column* Pointer to the column at `index`
   *---------------------------------------------------------------------------**/
  gdf_column* get_column(cudf::size_type index) {
    assert(index < _columns.size());
    return _columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer const of the column specified by an index.
   *
   * @param index The index of the desired column
   * @return gdf_column* Pointer to the column at `index`
   *---------------------------------------------------------------------------**/
  gdf_column const* get_column(cudf::size_type index) const {
    assert(index < _columns.size());
    return _columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of _columns in the table
   *
   *---------------------------------------------------------------------------**/
  cudf::size_type num_columns() const { return _columns.size(); }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of rows in the table
   *
   *---------------------------------------------------------------------------**/
  cudf::size_type num_rows() const {
    if (not _columns.empty()) { return this->get_column(0)->size; }
    return 0;
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns a new `table` containing the set of specified columns.
   *
   * @throws cudf::logic_error
   * If columns requested exceeds number of columns exist
   * @throws std::out_of_range
   * If the index of the column exceeds max column index in table
   *
   * @param column_indices Indices of the desired columns
   * @return table New table containing only the desired columns
   *---------------------------------------------------------------------------**/
  table select(std::vector<cudf::size_type> const& column_indices) const;

  /**---------------------------------------------------------------------------*
   * @brief Destroys the `gdf_column`s in the table.
   *
   * Free's each column's device memory and destroys the `gdf_column` object.
   *
   *---------------------------------------------------------------------------**/
  void destroy(void);

 private:
  std::vector<gdf_column*> _columns;  ///< Pointers to the wrapped columns
};

/**---------------------------------------------------------------------------*
 * @brief Returns vector of the dtypes of the columns in a table
 *
 * @param table The table to get the column dtypes from
 * @return std::vector<gdf_dtype>
 *---------------------------------------------------------------------------**/
std::vector<gdf_dtype> column_dtypes(cudf::table const& table);

/**---------------------------------------------------------------------------*
 * @brief Returns vector of the dtype_infos of the columns in a table
 *
 * @param table The table to get the column dtypes_infos from
 * @return std::vector<gdf_dtype_extra_info>
 *---------------------------------------------------------------------------**/
std::vector<gdf_dtype_extra_info> column_dtype_infos(cudf::table const& table);

/**---------------------------------------------------------------------------*
 * @brief Indicates if a table contains any null values.
 *
 * @param table The table to check for null values
 * @return true If the table contains one or more null values
 * @return false If the table contains zero null values
 *---------------------------------------------------------------------------**/
bool has_nulls(cudf::table const& table);

/**---------------------------------------------------------------------------*
 * @brief `table1` and `table2` are concatenated to return single table
 *
 * @throws cudf::logic_error
 * If number of rows mismatch
 *
 * @param table1 The table to be concatenated with `table2`
 * @param table2 The table to be concatenated with `table1`
 * @return A single table having all the columns from `table1` and `table2`
 * respectively in the same order.
 *---------------------------------------------------------------------------**/
cudf::table concat(cudf::table const& table1, cudf::table const& table2);

}  // namespace cudf

#endif
