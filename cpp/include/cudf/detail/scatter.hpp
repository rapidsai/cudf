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
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>

#include <memory>

namespace cudf {
namespace experimental {
namespace detail {

/**
 * @brief Scatters the rows of the source table into a copy of the target table
 * according to a scatter map.
 *
 * Scatters values from the source table into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table gets row
 * `i` of the source table. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of columns in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * A negative value `i` in the `scatter_map` is interpreted as `i+n`, where `n`
 * is the number of rows in the `target` table.
 * 
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * @throws `cudf::logic_error` if `check_bounds == true` and an index exists in
 * `scatter_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the target table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param source The input columns containing values to be scattered into the
 * target columns
 * @param scatter_map A non-nullable column of integral indices that maps the
 * rows in the source table to rows in the target table. The size must be equal
 * to or less than the number of elements in the source columns.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param mr The resource to use for all allocations
 * @param stream The stream to use for CUDA operations
 * @return Result of scattering values from source to target
 *---------------------------------------------------------------------------**/
std::unique_ptr<table> scatter(
    table_view const& source, column_view const& scatter_map,
    table_view const& target, bool check_bounds = false,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Scatters a row of scalar values into a copy of the target table
 * according to a scatter map.
 *
 * Scatters values from the source row into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table is
 * replaced by the source row. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of elements in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 * 
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * @throws `cudf::logic_error` if `check_bounds == true` and an index exists in
 * `scatter_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the target table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param source The input scalars containing values to be scattered into the
 * target columns
 * @param indices A non-nullable column of integral indices that indicate
 * the rows in the target table to be replaced by source.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param mr The resource to use for all allocations
 * @param stream The stream to use for CUDA operations
 * @return Result of scattering values from source to target
 *---------------------------------------------------------------------------**/
std::unique_ptr<table> scatter(
    std::vector<std::unique_ptr<scalar>> const& source, column_view const& indices,
    table_view const& target, bool check_bounds = false,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Scatters the rows of a table to `n` tables according to a partition map
 *
 * Copies the rows from the input table to new tables according to the table
 * indices given by partition_map. The number of output tables is one more than
 * the maximum value in `partition_map`.
 * 
 * Output table `i` in [0, n] is empty if `i` does not appear in partition_map.
 * output table will be empty.
 *
 * @throw cudf::logic_error when partition_map is a non-integer type
 * @throw cudf::logic_error when partition_map is larger than input
 * @throw cudf::logic_error when partition_map has nulls
 *
 * Example:
 * input:         [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *                 { 1,  2,  3,  4, null, 0, 2,  4,  6,  2}]
 * partition_map: {3,  4,  3,  1,  4,  4,  0,  1,  1,  1}
 * output:     {[{22}, {2}], 
 *              [{16, 24, 26, 28}, {4, 4, 6, 2}],
 *              [{}, {}],
 *              [{10, 14}, {1, 3}],
 *              [{12, 18, 20}, {2, null, 0}]}
 *
 * @param input Table of rows to be partitioned into a set of tables
 * tables according to `partition_map`
 * @param partition_map  Non-null column of integer values that map
 * each row in `input` table into one of the output tables
 * @param mr The resource to use for all allocations
 * @param stream The stream to use for CUDA operations
 *
 * @return A vector of tables containing the scattered rows of `input`.
 * `table` `i` contains all rows `j` from `input` where `partition_map[j] == i`.
 */
std::vector<std::unique_ptr<table>> scatter_to_tables(
    table_view const& input, column_view const& partition_map,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
