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

#include <copying/legacy/copy_range.cuh>

#include "scalar_factory.cuh"

namespace cudf {

void fill(gdf_column *column, gdf_scalar const& value, 
          gdf_index_type begin, gdf_index_type end)
{ 
  if (end != begin) { // otherwise no-op   
    validate(column);
    // TODO: once gdf_scalar supports string scalar values we can add support
    CUDF_EXPECTS(column->dtype != GDF_STRING_CATEGORY,
                 "cudf::fill() does not support GDF_STRING_CATEGORY columns");
    CUDF_EXPECTS(column->dtype == value.dtype, "Data type mismatch");
    detail::copy_range(column, detail::scalar_factory{value}, begin, end);
  }
}

}; // namespace cudf
