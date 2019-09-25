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

#ifndef COPYING_HPP
#define COPYING_HPP

#include "cudf.h"
#include "types.hpp"
#include <vector>

// Forward declaration
typedef struct CUstream_st* cudaStream_t;

namespace cudf {

/** ---------------------------------------------------------------------------*
* @brief Indicates when to allocate a mask, based on an existing mask.
* ---------------------------------------------------------------------------**/
enum mask_allocation_policy {
    NEVER, ///< Do not allocate a null mask, regardless of input
    RETAIN, ///< Allocate a null mask if the input contains one
    ALWAYS ///< Allocate a null mask, regardless of input
};

/*
 * Initializes and returns gdf_column of the same type as the input.
 * 
 * @param input The input column to emulate
 * @return gdf_column An unallocated column of same type as input
 */
gdf_column empty_like(gdf_column const& input);

/**
 * @brief Allocates a new column of the same size and type as the input.
 *
 * @param input The input column to emulate
 * @param mask_alloc Policy for allocating null mask. Defaults to RETAIN.
 * @param stream Optional stream in which to perform copies
 * @return gdf_column An allocated column of same size and type of input
 */
gdf_column allocate_like(gdf_column const& input, mask_allocation_policy mask_alloc = RETAIN,
                         cudaStream_t stream = 0);

/**
 * @brief Allocates a new column of the specified size and same type as the input.
 *
 * @param input The input column to emulate
 * @param size The size of the column to allocate in rows
 * @param mask_alloc Policy for allocating null mask. Defaults to RETAIN.
 * @param stream Optional stream in which to perform copies
 * @return gdf_column An allocated column of same size and type of input
 */
gdf_column allocate_like(gdf_column const& input, gdf_size_type size, mask_allocation_policy mask_alloc = RETAIN,
                         cudaStream_t stream = 0);


/**
 * @brief Creates a new column that is a copy of input
 * 
 * @param input The input column to copy
 * @param stream Optional stream in which to perform copies
 * @return gdf_column A copy of input
 */
gdf_column copy(gdf_column const& input, cudaStream_t stream = 0);

/**
 * @brief Creates a table of empty columns with the same types as the inputs
 *
 * Creates the `gdf_column` objects, but does not allocate any underlying device
 * memory for the column's data or bitmask.
 *
 * @note It is the caller's responsibility to delete the `gdf_column` object for
 * every column in the new table.
 *
 * @param t The input table to emulate
 * @return table A table of empty columns of same type as input
 */
table empty_like(table const& t);

/**
 * @brief Creates a table of columns with the same type and allocation size as
 * the input.
 *
 * Creates the `gdf_column` objects, and allocates underlying device memory for
 * each column matching the input columns
 *
 * @note It is the caller's responsibility to free each column's device memory
 * allocation in addition to deleting the `gdf_column` object for every column
 * in the new table.
 *
 * @param t The table to emulate
 * @param mask_alloc Policy for allocating null mask. Defaults to RETAIN.
 * @param stream Optional stream in which to perform allocations
 * @return table A table of columns with same type and allocation size as input
 */
table allocate_like(table const& t, mask_allocation_policy mask_alloc = RETAIN,
                    cudaStream_t stream = 0);

/**
 * @brief Creates a table of columns with the specified size and same type as
 * the input.
 *
 * Creates the `gdf_column` objects, and allocates underlying device memory for
 * each column matching the input columns
 *
 * @note It is the caller's responsibility to free each column's device memory
 * allocation in addition to deleting the `gdf_column` object for every column
 * in the new table.
 *
 * @param t The table to emulate
 * @param size The size of the columns to allocate
 * @param mask_alloc Policy for allocating null mask. Defaults to RETAIN.
 * @param stream Optional stream in which to perform allocations
 * @return table A table of columns with same type as @p t and specified @p size
 */
table allocate_like(table const& t, gdf_size_type size, mask_allocation_policy mask_alloc = RETAIN,
                    cudaStream_t stream = 0);


/**
 * @brief Creates a table of columns and deep copies the data from an input
 * table.
 *
 * @note It is the caller's responsibility to free each column's device memory
 * allocation in addition to deleting the `gdf_column` object for every column
 * in the new table.
 *
 * @param t The table to copy
 * @param stream Optional stream in which to perform allocations and copies
 * @return table A table that is an exact copy of @p t
 */
table copy(table const& t, cudaStream_t stream = 0);

/**
 * @brief Copies a range of elements from one column to another.
 * 
 * Copies N elements of @p in_column starting at @p in_begin to the N
 * elements of @p out_column starting at @p out_begin, where
 * N = (@p out_end - @p out_begin)
 * 
 * The datatypes of in_column and out_column must be the same.
 *
 * If the input and output columns are the same and ranges overlap, the
 * behavior is undefined.
 *
 * @param[out] out_column The preallocated column to copy into
 * @param[in] in_column The column to copy from
 * @param[in] out_begin The starting index of the output range
 * @param[in] out_end The index one past the end of the output range
 * @param[in] in_begin The starting index of the input range
 * 
 * @return void
 */
void copy_range(gdf_column *out_column, gdf_column const &in_column,
                gdf_index_type out_begin, gdf_index_type out_end, 
                gdf_index_type in_begin);

/**
 * @brief Creates a new `table` as if an in-place scatter from a `source` table 
 * was performed on the `target` table.
 *
 * It is the user's reponsibility to free the device memory allocated in the 
 * returned table `destination_table`.
 *
 * The `source_table` and the `target_table` must have equal numbers of columns.
 *
 * The datatypes between coresponding columns in the source and target columns 
 * must be the same.
 *
 * The number of rows in the scatter_map must equal the number of rows in
 * the source columns.
 *
 * If any index in scatter_map is outside the range of [0, target.num_rows()), 
 * the result is undefined.
 *
 * If the same index appears more than once in scatter_map, the result is
 * undefined.
 * 
 * A column in the output will only be nullable if: 
 * - Its corresponding column in `target` is nullable
 * - Its corresponding column in `source` has `null_count > 0` 
 *
 * @Param[in] source The columns whose rows will be scattered
 * @Param[in] scatter_map An array that maps rows in the input columns
 * to rows in the output columns.
 * @Param[in] target The table to copy and then perform an in-place scatter 
 * into the copy.
 * @return[out] The result of the scatter
 */
table scatter(table const& source, gdf_index_type const scatter_map[],
              table const& target);

/**
 * @brief Creates a new `table` as if scattering a set of `gdf_scalar`
 * values into the rows of a `target` table in-place.
 *
 * `data` and `valid` of a specific row of the target_column is kept 
 * unchanged if the `scatter_map` does not map to that row.
 * 
 * The datatypes between coresponding columns in the source and target
 * columns must be the same.
 *
 * If any index in scatter_map is outside the range of [0, num rows in
 * target_columns), the result is undefined.
 *
 * If the same index appears more than once in scatter_map, the result is
 * undefined.
 *
 * If the scalar is null (is_valid == false) and the target column does not
 * have a valid bitmask, the destination column will have a bitmask allocated.
 *
 * @Param[in] source The row to be scattered
 * @Param[in] scatter_map An array that maps to rows in the output columns.
 * @Param[in] target The table to copy and then perform an in-place scatter 
 * into the copy.
 * @return[out] The result of the scatter
 *
 */
table scatter(std::vector<gdf_scalar> const& source,
              gdf_index_type const scatter_map[],
              gdf_size_type num_scatter_rows, table const& target);

/**
 * @brief Gathers the rows (including null values) of a set of source columns
 * into a set of destination columns.
 * 
 * The two sets of columns must have equal numbers of columns.
 *
 * Gathers the rows of the source columns into the destination columns according
 * to a gather map such that row "i" in the destination columns will contain
 * row "gather_map[i]" from the source columns.
 *
 * The datatypes between coresponding columns in the source and destination
 * columns must be the same.
 *
 * The number of elements in the gather_map must equal the number of rows in the
 * destination columns.
 *
 * If any index in the gather_map is outside the range [0, num rows in
 * source_columns), the result is undefined.
 *
 * If the same index appears more than once in gather_map, the result is
 * undefined.
 *
 * @param[in] source_table The input columns whose rows will be gathered
 * @param[in] gather_map An array of indices that maps the rows in the source
 * columns to rows in the destination columns.
 * @param[out] destination_table A preallocated set of columns with a number
 * of rows equal in size to the number of elements in the gather_map that will
 * contain the rearrangement of the source columns based on the mapping. Can be
 * the same as `source_table` (in-place gather).
 *
 */
void gather(table const* source_table, gdf_index_type const gather_map[],
                 table* destination_table);

/**
 * @brief Slices a column (including null values) into a set of columns
 * according to a set of indices.
 *
 * The "slice" function divides part of the input column into multiple intervals
 * of rows using the indices values and it stores the intervals into the output
 * columns. Regarding the interval of indices, a pair of values are taken from
 * the indices array in a consecutive manner. The pair of indices are left-closed
 * and right-open.
 *
 * The pairs of indices in the array are required to comply with the following
 * conditions:
 * a, b belongs to Range[0, input column size]
 * a <= b, where the position of a is less or equal to the position of b.
  *
 * Exceptional cases for the indices array are:
 * When the values in the pair are equal, the function returns an empty column.
 * When the values in the pair are 'strictly decreasing', the outcome is
 * undefined.
 * When any of the values in the pair don't belong to the range[0, input column
 * size), the outcome is undefined.
 * When the indices array is empty, an empty vector of columns is returned.
 *
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * indices: {1, 3, 5, 9, 2, 4, 8, 8}
 * output:  {{12, 14}, {20, 22, 24, 26}, {14, 16}, {}}
 *
 * @param[in] input         The input column whose rows will be sliced.
 * @param[in] indices       An device array of indices that are used to take 'slices'
 * of the input column.
 * @param[in] num_indices   Number of indices in the indices array
 * @return  A std::vector of gdf_column*, each of which may have a different
 * number of rows. The number of rows in each column is equal to the difference
 * of two consecutive indices in the indices array.
 */
std::vector<gdf_column*> slice(gdf_column const &         input,
                               gdf_index_type const*      indices,
                               gdf_size_type              num_indices);

/**
 * @brief Splits a column (including null values) into a set of columns
 * according to a set of indices.
 *
 * The "split" function divides the input column into multiple intervals
 * of rows using the splits indices values and it stores the intervals into the
 * output columns. Regarding the interval of indices, a pair of values are taken
 * from the indices array in a consecutive manner. The pair of indices are
 * left-closed and right-open.
 *
 * The indices array ('splits') is require to be a monotonic non-decreasing set.
 * The indices in the array are required to comply with the following conditions:
 * a, b belongs to Range[0, input column size]
 * a <= b, where the position of a is less or equal to the position of b.
 *
 * The split function will take a pair of indices from the indices array
 * ('splits') in a consecutive manner. For the first pair, the function will
 * take the value 0 and the first element of the indices array. For the last pair,
 * the function will take the last element of the indices array and the size of
 * the input column.
 *
 * Exceptional cases for the indices array are:
 * When the values in the pair are equal, the function return an empty column.
 * When the values in the pair are 'strictly decreasing', the outcome is
 * undefined.
 * When any of the values in the pair don't belong to the range[0, input column
 * size), the outcome is undefined.
 * When the indices array is empty, an empty vector of columns is returned.
 *
 * The input columns may have different sizes. The number of
 * columns must be equal to the number of indices in the array plus one. 
 *
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * splits: {2, 5, 9}
 * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
 *
 * @param[in] input         The input column whose rows will be split.
 * @param[in] splits        An device array of indices that are used to divide
 * the input column into multiple columns.
 * @param[in] num_splits   Number of splits in the splits indices array
 * @return A std::vector of gdf_column*, each of which may have a different size
 * a different number of rows.
 */
std::vector<gdf_column*> split(gdf_column const &         input,
                               gdf_index_type const*      splits,
                               gdf_size_type              num_splits);

/**
 * @brief Scatters the rows of a table to `n` tables according to a scatter map
 *
 * Copies the rows from the input table to new
 * tables according to the table indices given by scatter map.
 * The number of output tables is one more than the maximum value in @p scatter_map.
 * If a value in [0,n] does not appear in scatter_map, then the corresponding
 * output table will be empty.
 *
 * `scatter_map` is a non-nullable column of `GDF_INT32` elements whose `size`
 * equals `input.num_rows()` and contains numbers in range of [0, n].
 *
 * Exceptional cases for the scatter_map column are:
 * @throws cudf::logic_error when `scatter_map.dtype != GDF_INT32`
 * @throws cudf::logic_error when `scatter_map.size != input.num_rows()`
 * @throws cudf::logic_error when `has_nulls(scatter_map) == true`
 *
 * Example:
 * input:       [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28}, 
 *               { 1,  2,  3,  4, null, 0, 2,  4,  6,  2}]
 * scatter_map:  { 3,  4,  3,  1,  4,  4,  0,  1,  1,  1}
 * output:     {[{22}, {2}], 
 *              [{16, 24, 26, 28}, {4, 4, 6, 2}], 
 *              [{}, {}], 
 *              [{10, 14}, {1, 3}], 
 *              [{12, 18, 20}, {2, null, 0}]}
 *
 * @param[in] input Table whose rows will be partitioned into a set of
 * tables according to `scatter_map` 
 * @param[in] scatter_map  Non-nullable column of `GDF_INT32` values that map
 * each row in `input` table into one of the output tables. 
 *
 * @return A std::vector of `table`s containing the scattered rows of `input`.
 * `table` `i` contains all rows `j` from `input` where `scatter_map[j] == i`. 
 *
 */
std::vector<cudf::table>
scatter_to_tables(cudf::table const& input, gdf_column const& scatter_map);
}  // namespace cudf

#endif  // COPYING_H
