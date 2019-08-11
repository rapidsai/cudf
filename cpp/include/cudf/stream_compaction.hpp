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

#include "cudf.h"
#include "types.hpp"


namespace cudf {

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
 * @brief Filters a table to remove null elements.
 *
 * Filters the rows of the input table considering only specified columns for
 * validity / null values.
 * 
 * Given an input table, row `i` from the input columns is copied to the
 * output if the same row `i` of @p keys has at leaast @p keep_threshold
 * non-null fields.
 *
 * This operation is stable: the input order is preserved in the output.
 * 
 * Note that the memory for the columns of the output table is allocated by this
 * function but must be freed by the caller when finished.
 *
 * Any non-nullable column in the input is treated as all non-null.
 *
 * @note if @p input.num_rows() is zero, or @p keys is empty or has no nulls,
 * there is no error, and an empty table is returned
 * 
 * @throws cudf::logic_error if @p keys is non-empty and keys.num_rows() is less
 * than input.num_rows()
 *
 * @param[in] input The input table to filter.
 * @param[in] keys The table of columns to check for nulls.
 * @param[in] keep_threshold The minimum number of non-null fields in a row
 *                           required to keep the row.
 * @return cudf::table Table containing all rows of the input table with at 
 *                     least @p keep_threshold non-null fields in @p keys.
 */
table drop_nulls(table const &input,
                 table const &keys,
                 gdf_size_type keep_threshold);

/**
 * @brief Filters a table to remove null elements.
 * 
 * @overload drop_nulls
 * 
 * Same as drop_nulls but defaults keep_threshold to the number of columns in 
 * @p keys.
 * 
 * @param[in] input The input table to filter.
 * @param[in] keys The table of columns to check for nulls.
 * @return cudf::table Table containing all rows of the input table without
 *                     nulls in the columns of @p keys.
 */
table drop_nulls(table const &input,
                 table const &keys);

/**
 * @brief Choices for drop_duplicates API for retainment of duplicate rows
 */
enum duplicate_keep_option {
  KEEP_FIRST = 0,   ///< Keeps first duplicate row and unique rows
  KEEP_LAST,        ///< Keeps last  duplicate row and unique rows
  KEEP_NONE         ///< Don't keep any duplicate rows, Keeps only unique rows
};

/**
 * @brief Create a new table without duplicate rows 
 *
 * Given an input table, each row is copied to output table if the corresponding
 * row of key column table is unique, where the definition of unique depends on the value of @p keep:
 * - KEEP_FIRST: only the first of a sequence of duplicate rows is copied
 * - KEEP_LAST: only the last of a sequence of duplicate rows is copied
 * - KEEP_NONE: no duplicate rows are copied 
 *
 * The input table and key columns table should have same number of rows.
 * Note that the memory for the output table columns is allocated by this function, so 
 * it must be freed by the caller when finished. 
 *
 * @param[in] input_table input table to copy only unique rows
 * @param[in] key_columns columns to consider to identify duplicate rows
 * @param[in] keep keep first entry, last entry, or no entries if duplicates found
 * @param[in] nulls_are_equal flag to denote nulls are equal if true,
 * nulls are not equal if false
 *
 * @return out_table with only unique rows
 */
cudf::table drop_duplicates(const cudf::table& input_table,
                            const cudf::table& key_columns,
                            const duplicate_keep_option keep,
                            const bool nulls_are_equal = true);
}  // namespace cudf

#endif
