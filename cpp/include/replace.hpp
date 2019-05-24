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

#ifndef REPLACE_HPP
#define REPLACE_HPP

#include "cudf.h"
#include "types.hpp"

// Forward declaration
typedef struct CUstream_st* cudaStream_t;

namespace cudf {

/**
 * @brief Replace elements from `col` according to the mapping `old_values` to
 *        `new_values`, that is, replace all `old_values[i]` present in `col`
 *        with `new_values[i]` and return a new gdf_column `out_col`.
 *
 * @param[in] col gdf_column with the data to be modified
 * @param[in] old_values gdf_column with the old values to be replaced
 * @param[in] new_values gdf_column with the new values
 *
 * @return gdf_column upon successful completion
 *
 */
gdf_column gdf_find_and_replace_all(gdf_column const &input_col,
                                    const gdf_column &old_values,
                                    const gdf_column &new_values);


} // namespace cudf

#endif // REPLACE_H