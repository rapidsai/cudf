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

#include <cudf.h>
#include <cassert>
#include <types.hpp>

#include <vector>
#include <cuda_runtime_api.h>

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
  table(gdf_column* cols[], gdf_size_type num_cols);

  /**---------------------------------------------------------------------------*
   * @brief Allocates and constructs a set of `gdf_column`s.
   *
   * Allocates an array of `gdf_column`s of the specified size and type.
   *
   * @note It is the caller's responsibility to free the array of gdf_columns
   *and their associated device memory.
   *
   * @note Does not support `GDF_TIMESTAMP` columns as this would require
   * passing in additional timestamp resolution information.
   *
   * @param[in] num_rows The size of each gdf_column
   * @param[in] dtypes The type of each column
   * @param[in] allocate_bitmasks If `true`, each column will be allocated an
   * appropriately sized bitmask
   *---------------------------------------------------------------------------**/
  table(gdf_size_type num_rows, std::vector<gdf_dtype> const& dtypes,
        bool allocate_bitmasks = false, bool all_valid = false,
        cudaStream_t stream = 0);

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
  gdf_column* get_column(gdf_index_type index) {
    assert(index < _columns.size());
    return _columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer const of the column specified by an index.
   *
   * @param index The index of the desired column
   * @return gdf_column* Pointer to the column at `index`
   *---------------------------------------------------------------------------**/
  gdf_column const* get_column(gdf_index_type index) const {
    assert(index < _columns.size());
    return _columns[index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of _columns in the table
   *
   *---------------------------------------------------------------------------**/
  gdf_size_type num_columns() const { return _columns.size(); }

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of rows in the table
   *
   *---------------------------------------------------------------------------**/
  gdf_size_type num_rows() const { return _num_rows; }

 private:
  std::vector<gdf_column*> _columns;  ///< Pointers to the wrapped columns
  gdf_size_type _num_rows{0};         ///< The number of elements in each column
};

}  // namespace cudf

#endif
