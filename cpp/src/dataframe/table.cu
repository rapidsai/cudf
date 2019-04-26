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

#include <cudf.h>
#include <types.hpp>

#include <utilities/error_utils.hpp>
#include <algorithm>
#include <cassert>


namespace cudf {

table::table(gdf_column* cols[], gdf_size_type num_cols)
      : columns{cols}, _num_columns{num_cols} {
    CUDF_EXPECTS(nullptr != cols[0], "Null input column");

    this->_num_rows = cols[0]->size;

    std::for_each(columns, columns + _num_columns, [this](gdf_column* col) {
      CUDF_EXPECTS(nullptr != col, "Null input column");
      CUDF_EXPECTS(_num_rows == col->size, "Column size mismatch");
    });
}


}  // namespace cudf


