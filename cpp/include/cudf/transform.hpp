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

#include <arrow/api.h>
#include <cudf/column/column.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/types.hpp>

#include <memory>

namespace cudf {
/**
 * @addtogroup transformation_transform
 * @{
 */

/**
 * @brief Creates a new column by applying a unary function against every
 * element of an input column.
 *
 * Computes:
 * `out[i] = F(in[i])`
 *
 * The output null mask is the same is the input null mask so if input[i] is
 * null then output[i] is also null
 *
 * @param input         An immutable view of the input column to transform
 * @param unary_udf     The PTX/CUDA string of the unary function to apply
 * @param outout_type   The output type that is compatible with the output type in the UDF
 * @param is_ptx        true: the UDF is treated as PTX code; false: the UDF is treated as CUDA code
 * @param mr            Device memory resource used to allocate the returned column's device memory
 * @return              The column resulting from applying the unary function to
 *                      every element of the input
 **/
std::unique_ptr<column> transform(
  column_view const& input,
  std::string const& unary_udf,
  data_type output_type,
  bool is_ptx,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Creates a null_mask from `input` by converting `NaN` to null and
 * preserving existing null values and also returns new null_count.
 *
 * @throws cudf::logic_error if `input.type()` is a non-floating type
 *
 * @param input         An immutable view of the input column of floating-point type
 * @param mr            Device memory resource used to allocate the returned bitmask.
 * @return A pair containing a `device_buffer` with the new bitmask and it's
 * null count obtained by replacing `NaN` in `input` with null.
 **/
std::pair<std::unique_ptr<rmm::device_buffer>, size_type> nans_to_nulls(
  column_view const& input, rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Creates a bitmask from a column of boolean elements.
 *
 * If element `i` in `input` is `true`, bit `i` in the resulting mask is set (`1`). Else,
 * if element `i` is `false` or null, bit `i` is unset (`0`).
 *
 *
 * @throws cudf::logic_error if `input.type()` is a non-boolean type
 *
 * @param input        Boolean elements to convert to a bitmask.
 * @param mr           Device memory resource used to allocate the returned bitmask.
 * @return A pair containing a `device_buffer` with the new bitmask and it's
 * null count obtained from input considering `true` represent `valid`/`1` and
 * `false` represent `invalid`/`0`.
 **/
std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input, rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Create `arrow::Table` from cudf table `input`
 *
 * Converts the `cudf::table_view` to `arrow::Table` with the provided
 * metadata `column_names`.
 *
 * @throws cudf::logic_error if `column_names` size doesn't match with number of columns.
 *
 * @param input table_view that needs to be converted to arrow Table
 * @param column_names Vector of column names for metadata of arrow Table
 * @param ar_mr arrow memory pool to allocate memory for arrow Table
 * @return arrow Table generated from given input cudf table
 **/
std::shared_ptr<arrow::Table> cudf_to_arrow(
  table_view input,
  std::vector<std::string> const& column_names = {},
  arrow::MemoryPool* ar_mr                     = arrow::default_memory_pool());

/**
 * @brief Create `cudf::table` from given arrow Table input
 *
 * @param input  arrow:Table that needs to be converted to `cudf::table`
 * @param mr           Device memory resource used to allocate `cudf::table`
 * @return cudf table generated from given arrow Table.
 **/

std::unique_ptr<table> arrow_to_cudf(
  std::shared_ptr<arrow::Table> input_table,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Encode the values of the given column as integers
 *
 * The encoded values are integers in the range [0, n), where `n`
 * is the number of distinct values in the input column.
 * The result column is such that keys[result[i]] == input[i],
 * where `keys` is the set of distinct values in `input` in sorted ascending order.
 * If nulls are present in the input column, they are encoded as the
 * integer `k`, where `k` is the number of distinct non-null values.
 *
 * Examples:
 * @code{.pseudo}
 * input: {'a', 'b', 'b', 'a'}
 * output: [{'a', 'b'}, {0, 1, 1, 0}]
 *
 * input: {1, 3, 1, 2, 9}
 * output: [{1, 2, 3, 9}, {0, 2, 0, 1, 3}]
 * @endcode
 *
 * @param input        Column containing values to be encoded
 * @param mr           Device memory resource used to allocate the returned columns's device memory
 * @return A pair containing the distinct values of the input column in sorter order,
 * and a column of integer indices representing the encoded values.
 */
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> encode(
  cudf::column_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of group
}  // namespace cudf
