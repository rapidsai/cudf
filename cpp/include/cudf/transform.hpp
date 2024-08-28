/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup transformation_transform
 * @{
 * @file
 * @brief Column APIs for transforming rows
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
 * @param output_type   The output type that is compatible with the output type in the UDF
 * @param is_ptx        true: the UDF is treated as PTX code; false: the UDF is treated as CUDA code
 * @param stream        CUDA stream used for device memory operations and kernel launches
 * @param mr            Device memory resource used to allocate the returned column's device memory
 * @return              The column resulting from applying the unary function to
 *                      every element of the input
 */
std::unique_ptr<column> transform(
  column_view const& input,
  std::string const& unary_udf,
  data_type output_type,
  bool is_ptx,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates a null_mask from `input` by converting `NaN` to null and
 * preserving existing null values and also returns new null_count.
 *
 * @throws cudf::logic_error if `input.type()` is a non-floating type
 *
 * @param input         An immutable view of the input column of floating-point type
 * @param stream        CUDA stream used for device memory operations and kernel launches
 * @param mr            Device memory resource used to allocate the returned bitmask
 * @return A pair containing a `device_buffer` with the new bitmask and it's
 * null count obtained by replacing `NaN` in `input` with null.
 */
std::pair<std::unique_ptr<rmm::device_buffer>, size_type> nans_to_nulls(
  column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute a new column by evaluating an expression tree on a table.
 *
 * This evaluates an expression over a table to produce a new column. Also called an n-ary
 * transform.
 *
 * @throws cudf::logic_error if passed an expression operating on table_reference::RIGHT.
 *
 * @param table The table used for expression evaluation
 * @param expr The root of the expression tree
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource
 * @return Output column
 */
std::unique_ptr<column> compute_column(
  table_view const& table,
  ast::expression const& expr,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates a bitmask from a column of boolean elements.
 *
 * If element `i` in `input` is `true`, bit `i` in the resulting mask is set (`1`). Else,
 * if element `i` is `false` or null, bit `i` is unset (`0`).
 *
 *
 * @throws cudf::logic_error if `input.type()` is a non-boolean type
 *
 * @param input        Boolean elements to convert to a bitmask
 * @param stream       CUDA stream used for device memory operations and kernel launches
 * @param mr           Device memory resource used to allocate the returned bitmask
 * @return A pair containing a `device_buffer` with the new bitmask and it's
 * null count obtained from input considering `true` represent `valid`/`1` and
 * `false` represent `invalid`/`0`.
 */
std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Encode the rows of the given table as integers
 *
 * The encoded values are integers in the range [0, n), where `n`
 * is the number of distinct rows in the input table.
 * The result table is such that `keys[result[i]] == input[i]`,
 * where `keys` is a table containing the distinct rows  in `input` in
 * sorted ascending order. Nulls, if any, are sorted to the end of
 * the `keys` table.
 *
 * Examples:
 * @code{.pseudo}
 * input: [{'a', 'b', 'b', 'a'}]
 * output: [{'a', 'b'}], {0, 1, 1, 0}
 *
 * input: [{1, 3, 1, 2, 9}, {1, 2, 1, 3, 5}]
 * output: [{1, 2, 3, 9}, {1, 3, 2, 5}], {0, 2, 0, 1, 3}
 * @endcode
 *
 * @param input Table containing values to be encoded
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return A pair containing the distinct row of the input table in sorter order,
 * and a column of integer indices representing the encoded rows.
 */
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> encode(
  cudf::table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Encodes `input` by generating a new column for each value in `categories` indicating the
 * presence of that value in `input`.
 *
 * The resulting per-category columns are returned concatenated as a single column viewed by a
 * `table_view`.
 *
 * The `i`th row of the `j`th column in the output table equals 1
 * if `input[i] == categories[j]`, and 0 otherwise.
 *
 * The `i`th row of the `j`th column in the output table equals 1
 * if input[i] == categories[j], and 0 otherwise.
 *
 * Examples:
 * @code{.pseudo}
 * input: [{'a', 'c', null, 'c', 'b'}]
 * categories: ['c', null]
 * output: [{0, 1, 0, 1, 0}, {0, 0, 1, 0, 0}]
 * @endcode
 *
 * @throws cudf::logic_error if input and categories are of different types.
 *
 * @param input Column containing values to be encoded
 * @param categories Column containing categories
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return A pair containing the owner to all encoded data and a table view into the data
 */
std::pair<std::unique_ptr<column>, table_view> one_hot_encode(
  column_view const& input,
  column_view const& categories,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates a boolean column from given bitmask.
 *
 * Returns a `bool` for each bit in `[begin_bit, end_bit)`. If bit `i` in least-significant bit
 * numbering is set (1), then element `i` in the output is `true`, otherwise `false`.
 *
 * @throws cudf::logic_error if `bitmask` is null and end_bit-begin_bit > 0
 * @throws cudf::logic_error if begin_bit > end_bit
 *
 * Examples:
 * @code{.pseudo}
 * input: {0b10101010}
 * output: [{false, true, false, true, false, true, false, true}]
 * @endcode
 *
 * @param bitmask A device pointer to the bitmask which needs to be converted
 * @param begin_bit position of the bit from which the conversion should start
 * @param end_bit position of the bit before which the conversion should stop
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return A boolean column representing the given mask from [begin_bit, end_bit)
 */
std::unique_ptr<column> mask_to_bools(
  bitmask_type const* bitmask,
  size_type begin_bit,
  size_type end_bit,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns an approximate cumulative size in bits of all columns in the `table_view` for
 * each row.
 *
 * This function counts bits instead of bytes to account for the null mask which only has one
 * bit per row.
 *
 * Each row in the returned column is the sum of the per-row size for each column in
 * the table.
 *
 * In some cases, this is an inexact approximation. Specifically, columns of lists and strings
 * require N+1 offsets to represent N rows. It is up to the caller to calculate the small
 * additional overhead of the terminating offset for any group of rows being considered.
 *
 * This function returns the per-row sizes as the columns are currently formed. This can
 * end up being larger than the number you would get by gathering the rows. Specifically,
 * the push-down of struct column validity masks can nullify rows that contain data for
 * string or list columns. In these cases, the size returned is conservative:
 *
 * row_bit_count(column(x)) >= row_bit_count(gather(column(x)))
 *
 * @param t The table view to perform the computation on
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return A 32-bit integer column containing the per-row bit counts
 */
std::unique_ptr<column> row_bit_count(
  table_view const& t,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns an approximate cumulative size in bits of all columns in the `table_view` for
 * each segment of rows.
 *
 * This is similar to counting bit size per row for the input table in `cudf::row_bit_count`,
 * except that row sizes are accumulated by segments.
 *
 * Currently, only fixed-length segments are supported. In case the input table has number of rows
 * not divisible by `segment_length`, its last segment is considered as shorter than the others.
 *
 * @throw std::invalid_argument if the input `segment_length` is non-positive or larger than the
 * number of rows in the input table.
 *
 * @param t The table view to perform the computation on
 * @param segment_length The number of rows in each segment for which the total size is computed
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return A 32-bit integer column containing the bit counts for each segment of rows
 */
std::unique_ptr<column> segmented_row_bit_count(
  table_view const& t,
  size_type segment_length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
