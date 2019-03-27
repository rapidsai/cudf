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

#ifndef STREAM_COMPACTION_HPP
#define STREAM_COMPACTION_HPP

#include "cudf.h"

namespace cudf {

/**
 * @brief Filters a column using a column of boolean values as a mask.
 *
 * Given an input column and a mask column, an element `i` from the input column
 * is copied to the output if the corresponding element `i` in the mask is
 * non-null and `true`.
 *
 * The input and mask columns must be of equal size.
 *
 * The output column will have size equal to the number of elements in
 * boolean_mask that are both non-null and `true`. Note that the output column
 * memory is allocated by this function but must be freed by the caller when
 * finished.
 *
 * @param[in] input The input column to filter
 * @param[in] boolean_mask A column of type GDF_BOOL used as a mask to filter
 * the input column corresponding index passes the filter.
 * @param[out] output The output column containing the filtered result
 * @return gdf_column Column containing copy of all elements of @p input passing
 * the filter defined by @p boolean_mask.
 */
gdf_column apply_boolean_mask(gdf_column const *input,
                              gdf_column const *boolean_mask);
}  // namespace cudf

#endif