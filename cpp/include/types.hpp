/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cassert>
#include "cudf.h"

namespace cudf {

/**
 * @brief A wrapper for a set of gdf_columns all with equal number of rows.
 *
 */
struct table {
  table(gdf_column* cols[], gdf_size_type num_cols)
      : columns{cols}, num_columns{num_cols} {
    assert(nullptr != cols[0]);

    gdf_size_type const num_rows{cols[0]->size};

    std::for_each(columns, columns + num_columns, [num_rows](gdf_column* col) {
      assert(nullptr != col);
      assert(num_rows == col->size);
    });
  }

  gdf_column** data() const { return columns; }

  gdf_column** begin() const { return columns; }

  gdf_column** end() const { return columns + num_columns; }

  gdf_column* get_column(gdf_index_type index) { return columns[index]; }

  gdf_size_type size() const { return num_columns; }

 private:
  gdf_column** columns;            /**< The set of gdf_columns*/
  gdf_size_type const num_columns; /**< The number of columns in the set */
};

/**---------------------------------------------------------------------------*
 * @brief Type-erased column of possibly null elements residing in GPU device
 * memory.
 *
 *---------------------------------------------------------------------------**/
struct column {
  column(gdf_column const& col)
      : data{col.data},
        bitmask{col.valid},
        size{col.size},
        dtype{col.dtype},
        null_count{col.null_count},
        dtype_info{col.dtype_info} {}

 private:
  void* data;  ///< Type-erased device buffer containing elements of the column

  // TODO Use BitMask class instead
  gdf_valid_type*
      null_mask;  ///< Optional device-allocated bitmask where bit `i` indicates
                  ///< if element `i` is null or non-null. 0 indicates null, 1
                  ///< indicates non-null. If the bitmask does not exist, it is
                  ///< assumed all elements are valid.

  gdf_size_type size;  ///< The number of elements in the column

  // TODO make gdf_dtype a enum class
  gdf_dtype dtype;  ///< Runtime type information of column elements

  gdf_size_type null_count;  ///< The count of null elements in the column.
                             ///< If the bitmask does not exist, the null_count
                             ///< must be zero

  // TODO C++ abstraction for this?
  gdf_dtype_extra_info
      dtype_info;  ///< Additional type information, e.g., timestamp resolution
}

}  // namespace cudf

#endif