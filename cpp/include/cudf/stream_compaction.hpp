/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup reorder_compact
 * @{
 * @file
 * @brief Column APIs for filtering rows
 */

/**
 * @brief Filters a table to remove null elements with threshold count.
 *
 * Filters the rows of the `input` considering specified columns indicated in
 * `keys` for validity / null values.
 *
 * Given an input table_view, row `i` from the input columns is copied to
 * the output if the same row `i` of @p keys has at least @p keep_threshold
 * non-null fields.
 *
 * This operation is stable: the input order is preserved in the output.
 *
 * Any non-nullable column in the input is treated as all non-null.
 *
 * @code{.pseudo}
 *          input   {col1: {1, 2,    3,    null},
 *                   col2: {4, 5,    null, null},
 *                   col3: {7, null, null, null}}
 *          keys = {0, 1, 2} // All columns
 *          keep_threshold = 2
 *
 *          output {col1: {1, 2}
 *                  col2: {4, 5}
 *                  col3: {7, null}}
 * @endcode
 *
 * @note if @p input.num_rows() is zero, or @p keys is empty or has no nulls,
 * there is no error, and an empty `table` is returned
 *
 * @param[in] input The input `table_view` to filter
 * @param[in] keys  vector of indices representing key columns from `input`
 * @param[in] keep_threshold The minimum number of non-null fields in a row
 *                           required to keep the row.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @return Table containing all rows of the `input` with at least @p
 * keep_threshold non-null fields in @p keys.
 */
std::unique_ptr<table> drop_nulls(
  table_view const& input,
  std::vector<size_type> const& keys,
  cudf::size_type keep_threshold,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Filters a table to remove null elements.
 *
 * Filters the rows of the `input` considering specified columns indicated in
 * `keys` for validity / null values.
 *
 * @code{.pseudo}
 *          input   {col1: {1, 2,    3,    null},
 *                   col2: {4, 5,    null, null},
 *                   col3: {7, null, null, null}}
 *          keys = {0, 1, 2} //All columns
 *
 *          output {col1: {1}
 *                  col2: {4}
 *                  col3: {7}}
 * @endcode
 *
 * Same as drop_nulls but defaults keep_threshold to the number of columns in
 * @p keys.
 *
 * @param[in] input The input `table_view` to filter
 * @param[in] keys  vector of indices representing key columns from `input`
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @return Table containing all rows of the `input` without nulls in the columns
 * of @p keys.
 */
std::unique_ptr<table> drop_nulls(
  table_view const& input,
  std::vector<size_type> const& keys,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Filters a table to remove NANs with threshold count.
 *
 * Filters the rows of the `input` considering specified columns indicated in
 * `keys` for NANs. These key columns must be of floating-point type.
 *
 * Given an input table_view, row `i` from the input columns is copied to
 * the output if the same row `i` of @p keys has at least @p keep_threshold
 * non-NAN elements.
 *
 * This operation is stable: the input order is preserved in the output.
 *
 * @code{.pseudo}
 *          input   {col1: {1.0, 2.0, 3.0, NAN},
 *                   col2: {4.0, null, NAN, NAN},
 *                   col3: {7.0, NAN, NAN, NAN}}
 *          keys = {0, 1, 2} // All columns
 *          keep_threshold = 2
 *
 *          output {col1: {1.0, 2.0}
 *                  col2: {4.0, null}
 *                  col3: {7.0, NAN}}
 * @endcode
 *
 * @note if @p input.num_rows() is zero, or @p keys is empty,
 * there is no error, and an empty `table` is returned
 *
 * @throws cudf::logic_error if The `keys` columns are not floating-point type.
 *
 * @param[in] input The input `table_view` to filter
 * @param[in] keys  vector of indices representing key columns from `input`
 * @param[in] keep_threshold The minimum number of non-NAN elements in a row
 *                           required to keep the row.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @return Table containing all rows of the `input` with at least @p
 * keep_threshold non-NAN elements in @p keys.
 */
std::unique_ptr<table> drop_nans(
  table_view const& input,
  std::vector<size_type> const& keys,
  cudf::size_type keep_threshold,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Filters a table to remove NANs.
 *
 * Filters the rows of the `input` considering specified columns indicated in
 * `keys` for NANs. These key columns must be of floating-point type.
 *
 * @code{.pseudo}
 *          input   {col1: {1.0, 2.0, 3.0, NAN},
 *                   col2: {4.0, null, NAN, NAN},
 *                   col3: {null, NAN, NAN, NAN}}
 *          keys = {0, 1, 2} // All columns
 *          keep_threshold = 2
 *
 *          output {col1: {1.0}
 *                  col2: {4.0}
 *                  col3: {null}}
 * @endcode
 *
 * Same as drop_nans but defaults keep_threshold to the number of columns in
 * @p keys.
 *
 * @param[in] input The input `table_view` to filter
 * @param[in] keys  vector of indices representing key columns from `input`
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @return Table containing all rows of the `input` without NANs in the columns
 * of @p keys.
 */
std::unique_ptr<table> drop_nans(
  table_view const& input,
  std::vector<size_type> const& keys,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Filters `input` using `boolean_mask` of boolean values as a mask.
 *
 * Given an input `table_view` and a mask `column_view`, an element `i` from
 * each column_view of the `input` is copied to the corresponding output column
 * if the corresponding element `i` in the mask is non-null and `true`.
 * This operation is stable: the input order is preserved.
 *
 * @note if @p input.num_rows() is zero, there is no error, and an empty table
 * is returned.
 *
 * @throws cudf::logic_error if `input.num_rows() != boolean_mask.size()`.
 * @throws cudf::logic_error if `boolean_mask` is not `type_id::BOOL8` type.
 *
 * @param[in] input The input table_view to filter
 * @param[in] boolean_mask A nullable column_view of type type_id::BOOL8 used
 * as a mask to filter the `input`.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @return Table containing copy of all rows of @p input passing
 * the filter defined by @p boolean_mask.
 */
std::unique_ptr<table> apply_boolean_mask(
  table_view const& input,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Choices for drop_duplicates API for retainment of duplicate rows
 */
enum class duplicate_keep_option {
  KEEP_ANY = 0,  ///< Keep an unspecified occurrence
  KEEP_FIRST,    ///< Keep first occurrence
  KEEP_LAST,     ///< Keep last occurrence
  KEEP_NONE      ///< Keep no (remove all) occurrences of duplicates
};

/**
 * @brief Create a new table with consecutive duplicate rows removed.
 *
 * Given an `input` table_view, each row is copied to the output table to create a set of distinct
 * rows. If there are duplicate rows, which row is copied depends on the `keep` parameter.
 *
 * The order of rows in the output table remains the same as in the input.
 *
 * A row is distinct if there are no equivalent rows in the table. A row is unique if there is no
 * adjacent equivalent row. That is, keeping distinct rows removes all duplicates in the
 * table/column, while keeping unique rows only removes duplicates from consecutive groupings.
 *
 * Performance hint: if the input is pre-sorted, `cudf::unique` can produce an equivalent result
 * (i.e., same set of output rows) but with less running time than `cudf::distinct`.
 *
 * @throws cudf::logic_error if the `keys` column indices are out of bounds in the `input` table.
 *
 * @param[in] input           input table_view to copy only unique rows
 * @param[in] keys            vector of indices representing key columns from `input`
 * @param[in] keep            keep any, first, last, or none of the found duplicates
 * @param[in] nulls_equal     flag to denote nulls are equal if null_equality::EQUAL, nulls are not
 *                            equal if null_equality::UNEQUAL
 * @param[in] stream          CUDA stream used for device memory operations and kernel launches
 * @param[in] mr              Device memory resource used to allocate the returned table's device
 *                            memory
 *
 * @return Table with unique rows from each sequence of equivalent rows as specified by `keep`
 */
std::unique_ptr<table> unique(
  table_view const& input,
  std::vector<size_type> const& keys,
  duplicate_keep_option keep,
  null_equality nulls_equal         = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a new table without duplicate rows.
 *
 * Given an `input` table_view, each row is copied to the output table to create a set of distinct
 * rows. If there are duplicate rows, which row is copied depends on the `keep` parameter.
 *
 * The order of rows in the output table is not specified.
 *
 * Performance hint: if the input is pre-sorted, `cudf::unique` can produce an equivalent result
 * (i.e., same set of output rows) but with less running time than `cudf::distinct`.
 *
 * @param input The input table
 * @param keys Vector of indices indicating key columns in the `input` table
 * @param keep Copy any, first, last, or none of the found duplicates
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether NaN elements should be considered as equal
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table
 * @return Table with distinct rows in an unspecified order
 */
std::unique_ptr<table> distinct(
  table_view const& input,
  std::vector<size_type> const& keys,
  duplicate_keep_option keep        = duplicate_keep_option::KEEP_ANY,
  null_equality nulls_equal         = null_equality::EQUAL,
  nan_equality nans_equal           = nan_equality::ALL_EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a column of indices of all distinct rows in the input table.
 *
 * Given an `input` table_view, an output vector of all row indices of the distinct rows is
 * generated. If there are duplicate rows, which index is kept depends on the `keep` parameter.
 *
 * @param input The input table
 * @param keep Get index of any, first, last, or none of the found duplicates
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether NaN elements should be considered as equal
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return Column containing the result indices
 */
std::unique_ptr<column> distinct_indices(
  table_view const& input,
  duplicate_keep_option keep        = duplicate_keep_option::KEEP_ANY,
  null_equality nulls_equal         = null_equality::EQUAL,
  nan_equality nans_equal           = nan_equality::ALL_EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a new table without duplicate rows, preserving input order.
 *
 * Given an `input` table_view, each row is copied to the output table to create a set of distinct
 * rows. The input row order is preserved. If there are duplicate rows, which row is copied depends
 * on the `keep` parameter.
 *
 * This API produces the same output rows as `cudf::distinct`, but with input order preserved.
 *
 * Note that when `keep` is `KEEP_ANY`, the choice of which duplicate row to keep is arbitrary, but
 * the returned table will retain the input order. That is, if the key column contained `1, 2, 1`
 * with another values column `3, 4, 5`, the result could contain values `3, 4` or `4, 5` but not
 * `4, 3` or `5, 4`.
 *
 * @param input The input table
 * @param keys Vector of indices indicating key columns in the `input` table
 * @param keep Copy any, first, last, or none of the found duplicates
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether NaN elements should be considered as equal
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table
 * @return Table with distinct rows, preserving input order
 */
std::unique_ptr<table> stable_distinct(
  table_view const& input,
  std::vector<size_type> const& keys,
  duplicate_keep_option keep        = duplicate_keep_option::KEEP_ANY,
  null_equality nulls_equal         = null_equality::EQUAL,
  nan_equality nans_equal           = nan_equality::ALL_EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Count the number of consecutive groups of equivalent rows in a column.
 *
 * If `null_handling` is null_policy::EXCLUDE and `nan_handling` is  nan_policy::NAN_IS_NULL, both
 * `NaN` and `null` values are ignored. If `null_handling` is null_policy::EXCLUDE and
 * `nan_handling` is nan_policy::NAN_IS_VALID, only `null` is ignored, `NaN` is considered in count.
 *
 * `null`s are handled as equal.
 *
 * @param[in] input The column_view whose consecutive groups of equivalent rows will be counted
 * @param[in] null_handling flag to include or ignore `null` while counting
 * @param[in] nan_handling flag to consider `NaN==null` or not
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return number of consecutive groups of equivalent rows in the column
 */
cudf::size_type unique_count(column_view const& input,
                             null_policy null_handling,
                             nan_policy nan_handling,
                             rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Count the number of consecutive groups of equivalent rows in a table.
 *
 * @param[in] input Table whose consecutive groups of equivalent rows will be counted
 * @param[in] nulls_equal flag to denote if null elements should be considered equal
 *            nulls are not equal if null_equality::UNEQUAL.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return number of consecutive groups of equivalent rows in the column
 */
cudf::size_type unique_count(table_view const& input,
                             null_equality nulls_equal    = null_equality::EQUAL,
                             rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Count the distinct elements in the column_view.
 *
 * If `nulls_equal == nulls_equal::UNEQUAL`, all `null`s are distinct.
 *
 * Given an input column_view, number of distinct elements in this column_view is returned.
 *
 * If `null_handling` is null_policy::EXCLUDE and `nan_handling` is  nan_policy::NAN_IS_NULL, both
 * `NaN` and `null` values are ignored. If `null_handling` is null_policy::EXCLUDE and
 * `nan_handling` is nan_policy::NAN_IS_VALID, only `null` is ignored, `NaN` is considered in
 * distinct count.
 *
 * `null`s are handled as equal.
 *
 * @param[in] input The column_view whose distinct elements will be counted
 * @param[in] null_handling flag to include or ignore `null` while counting
 * @param[in] nan_handling flag to consider `NaN==null` or not
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return number of distinct rows in the table
 */
cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Count the distinct rows in a table.
 *
 * @param[in] input Table whose distinct rows will be counted
 * @param[in] nulls_equal flag to denote if null elements should be considered equal.
 *            nulls are not equal if null_equality::UNEQUAL.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return number of distinct rows in the table
 */
cudf::size_type distinct_count(table_view const& input,
                               null_equality nulls_equal    = null_equality::EQUAL,
                               rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Creates a new column by applying a filter function against every
 * element of the input columns.
 *
 * Null values in the input columns are considered as not matching the filter.
 *
 * Computes:
 * `out[i]... = predicate(columns[i]... ) ? (columns[i]...): not-applied`.
 *
 * Note that for every scalar in `columns` (columns of size 1), `columns[i] ==
 * input[0]`
 *
 *
 * @throws std::invalid_argument if any of the input columns have different sizes (except scalars of
 * size 1)
 * @throws std::invalid_argument if `output_type` or any of the inputs are not fixed-width or string
 * types
 * @throws cudf::logic_error if JIT is not supported by the runtime
 * @throws std::invalid_argument if the size of `copy_mask` does not match the number of input
 * columns
 *
 * The size of the resulting column is the size of the largest column.
 *
 * @param columns       Immutable views of the columns to filter
 * @param predicate_udf The PTX/CUDA string of the transform function to apply
 * @param is_ptx        true: the UDF is treated as PTX code; false: the UDF is treated as CUDA code
 * @param user_data     User-defined device data to pass to the UDF.
 * @param copy_mask     Optional vector of booleans indicating which columns to copy from the input
 *                      columns to the output. If not provided, all columns are copied.
 * @param stream        CUDA stream used for device memory operations and kernel launches
 * @param mr            Device memory resource used to allocate the returned column's device memory
 * @return              The filtered target columns
 */
std::vector<std::unique_ptr<column>> filter(
  std::vector<column_view> const& columns,
  std::string const& predicate_udf,
  bool is_ptx,
  std::optional<void*> user_data             = std::nullopt,
  std::optional<std::vector<bool>> copy_mask = std::nullopt,
  rmm::cuda_stream_view stream               = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr          = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace CUDF_EXPORT cudf
