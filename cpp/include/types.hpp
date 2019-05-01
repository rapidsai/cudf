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

#include "cudf.h"

/**---------------------------------------------------------------------------*
 * @file types.hpp
 * @brief Type declarations for libcudf.
 * 
*---------------------------------------------------------------------------**/


namespace cudf {
// Forward declaration
struct table;


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


