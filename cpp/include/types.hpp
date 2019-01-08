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

}  // namespace cudf

#endif