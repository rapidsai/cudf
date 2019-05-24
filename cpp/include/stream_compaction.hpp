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

#ifndef STREAM_COMPACTION_HPP
#define STREAM_COMPACTION_HPP

#include <cudf.h>
#include <types.hpp>

namespace cudf {

/**
 * @brief Filters a column using a column of boolean values as a mask.
 *
 * Given an input column and a mask column, an element `i` from the input column
 * is copied to the output if the corresponding element `i` in the mask is
 * non-null and `true`. This operation is stable: the input order is preserved.
 *
 * The input and mask columns must be of equal size.
 *
 * The output column has size equal to the number of elements in boolean_mask 
 * that are both non-null and `true`. Note that the output column memory is 
 * allocated by this function but must be freed by the caller when finished.
 * 
 * @note that the @p boolean_mask may have just boolean data (no valid bitmask), 
 * or just a valid bitmask (no boolean data), or it may have both. The filter
 * adapts to these three situations.
 * 
 * @note if @p input.size is zero, there is no error, and an empty column is 
 * returned.
 * 
 * @param[in] input The input column to filter
 * @param[in] boolean_mask A column of type GDF_BOOL8 used as a mask to filter
 * the input column corresponding index passes the filter.
  * @return gdf_column Column containing copy of all elements of @p input passing
 * the filter defined by @p boolean_mask.
 */
gdf_column apply_boolean_mask(gdf_column const &input,
                              gdf_column const &boolean_mask);

/**
 * @brief Filters a table using a column of boolean values as a mask.
 *
 * Given an input table and a mask column, an element `i` from each column of
 * the input table is copied to the corresponding output column if the
 * corresponding element `i` in the mask is non-null and `true`. This operation
 * is stable: the input order is preserved.
 *
 * The input and mask columns must be of equal size (number of rows).
 *
 * The output table has number of rows equal to the number of elements in
 * boolean_mask that are both non-null and `true`. Note that the output table
 * memory is allocated by this function but must be freed by the caller when
 * finished.
 *
 * @note that the @p boolean_mask may have just boolean data (no valid bitmask),
 * or just a valid bitmask (no boolean data), or it may have both. The filter
 * adapts to these three situations.
 *
 * @note if @p input.num_rows() is zero, there is no error, and an empty table
 * is returned.
 * 
 * @param[in] input The input table to filter
 * @param[in] boolean_mask A column of type GDF_BOOL8 used as a mask to filter
 * the input column corresponding index passes the filter.
  * @return cudf::table Table containing copy of all rows of @p input passing
 * the filter defined by @p boolean_mask.
 */
table apply_boolean_mask(table const &input,
                         gdf_column const &boolean_mask);

/**
 * @brief Filters a column to remove null elements.
 *
 * Given an input column an element `i` from the input column is copied to the
 * output if the corresponding element `i` in the input's valid bitmask is
 * non-null.
 *
 * The output column has size equal to the number of elements in boolean_mask
 * that are both non-null and `true`. Note that the output column memory is
 * allocated by this function but must be freed by the caller when finished.
 *
 * If the input column is not nullable, this function just copies the input
 * to the output.
 *
 * * @note if @p input.size is zero, there is no error, and an empty column is
 * returned.
 *
 * @param[in] input The input column to filter
 * @return gdf_column Column containing copy of all non-null elements of @p input.
 */
gdf_column drop_nulls(gdf_column const &input);
}  // namespace cudf

#endif
