/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

namespace ast {
struct expression;
}

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

// Forward declaration
namespace detail {
struct approx_distinct_count;
}

/**
 * @brief Object-oriented HyperLogLog sketch for approximate distinct counting.
 *
 * This class provides an object-oriented interface to HyperLogLog sketches, allowing
 * incremental addition of data and cardinality estimation.
 *
 * The implementation uses XXHash64 to hash table rows into 64-bit values, which are
 * then added to the HyperLogLog sketch without additional hashing (identity function).
 *
 * @par HyperLogLog Precision Parameter
 * The precision parameter (p) is the number of bits used to index into the register array.
 * It determines the number of registers (m = 2^p) in the HLL sketch:
 * - Memory usage: 2^p bytes (m registers of 1 byte each)
 * - Standard error: 1.04 / sqrt(m) = 1.04 / sqrt(2^p)
 *
 * Common precision values:
 * - p=10: m=1,024 registers, ~3.2% standard error, 1KB memory
 * - p=12 (default): m=4,096 registers, ~1.6% standard error, 4KB memory
 * - p=14: m=16,384 registers, ~0.8% standard error, 16KB memory
 * - p=16: m=65,536 registers, ~0.4% standard error, 64KB memory
 *
 * Valid range: p ∈ [4, 18]. Higher precision provides better accuracy but uses more memory.
 *
 * Example usage:
 * @code{.cpp}
 *   auto adc = cudf::approx_distinct_count(table1);
 *   auto count1 = adc.estimate();
 *
 *   adc.add(table2);
 *   auto count2 = adc.estimate();
 * @endcode
 */
class approx_distinct_count {
 public:
  using impl_type = cudf::detail::approx_distinct_count;  ///< Implementation type

  /**
   * @brief Construct an approximate distinct count sketch from a table.
   *
   * @param input Table whose rows will be added to the sketch
   * @param precision The precision parameter for HyperLogLog (4-18). Higher precision gives
   *                  better accuracy but uses more memory. Default is 12.
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls (default: `EXCLUDE`)
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL` (default: `NAN_IS_NULL`)
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  approx_distinct_count(table_view const& input,
                        std::int32_t precision       = 12,
                        null_policy null_handling    = null_policy::EXCLUDE,
                        nan_policy nan_handling      = nan_policy::NAN_IS_NULL,
                        rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Construct an approximate distinct count sketch from serialized sketch bytes.
   *
   * This constructor enables distributed distinct counting by allowing sketches to be
   * constructed from serialized data. The sketch data is copied into the newly created
   * object, which then owns its own independent storage.
   *
   * @warning The precision parameter must match the precision used to create the original
   * sketch. The size of the sketch span must be exactly 2^precision bytes. Providing
   * an incompatible sketch will produce incorrect results or errors.
   *
   * @param sketch_span The serialized sketch bytes to reconstruct from
   * @param precision The precision parameter that was used to create the sketch (4-18)
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  approx_distinct_count(cuda::std::span<cuda::std::byte> sketch_span,
                        std::int32_t precision,
                        rmm::cuda_stream_view stream = cudf::get_default_stream());

  ~approx_distinct_count();

  approx_distinct_count(approx_distinct_count const&)            = delete;
  approx_distinct_count& operator=(approx_distinct_count const&) = delete;
  /** @brief Default move constructor */
  approx_distinct_count(approx_distinct_count&&) = default;
  /**
   * @brief Default move assignment operator
   * @return Reference to this object
   */
  approx_distinct_count& operator=(approx_distinct_count&&) = default;

  /**
   * @brief Add rows from a table to the sketch.
   *
   * @param input Table whose rows will be added
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls (default: `EXCLUDE`)
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL` (default: `NAN_IS_NULL`)
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void add(table_view const& input,
           null_policy null_handling    = null_policy::EXCLUDE,
           nan_policy nan_handling      = nan_policy::NAN_IS_NULL,
           rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Merge another sketch into this sketch.
   *
   * After merging, this sketch will contain the combined distinct count estimate of both sketches.
   *
   * @param other The sketch to merge into this sketch
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void merge(approx_distinct_count const& other,
             rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Merge a sketch from raw bytes into this sketch.
   *
   * This allows merging sketches that have been serialized or created elsewhere, enabling
   * distributed distinct counting scenarios.
   *
   * @warning It is the caller's responsibility to ensure that the provided sketch span was created
   * with the same approx_distinct_count configuration (precision, null/NaN handling, etc.) as this
   * sketch. Merging incompatible sketches will produce incorrect results.
   *
   * @param sketch_span The sketch bytes to merge into this sketch
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void merge(cuda::std::span<cuda::std::byte> sketch_span,
             rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Get the raw sketch bytes for serialization or external merging.
   *
   * The returned span provides access to the internal sketch storage.
   * This can be used to serialize the sketch, transfer it between processes,
   * or merge it with other sketches using the span-based merge API.
   *
   * @return A span view of the sketch bytes
   */
  [[nodiscard]] cuda::std::span<cuda::std::byte> sketch() noexcept;

  /**
   * @brief Estimate the approximate number of distinct rows in the sketch.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Approximate number of distinct rows
   */
  [[nodiscard]] cudf::size_type estimate(
    rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

 private:
  std::unique_ptr<impl_type> _impl;
};

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
 * @throws std::invalid_argument if the output or any of the inputs are not fixed-width or string
 * types
 * @throws cudf::logic_error if JIT is not supported by the runtime
 * @throws std::invalid_argument if the size of `copy_mask` does not match the number of input
 * columns
 *
 * The size of the resulting column is the size of the largest column.
 *
 * @param predicate_columns Immutable views of the predicate columns
 * @param predicate_udf The PTX/CUDA string of the transform function to apply
 * @param filter_columns Immutable view of the columns to be filtered
 * @param is_ptx true: the UDF is treated as PTX code; false: the UDF is treated as CUDA code
 * @param user_data User-defined device data to pass to the UDF.
 * @param is_null_aware Signifies the UDF will receive row inputs as optional values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The filtered target columns
 */
std::vector<std::unique_ptr<column>> filter(
  std::vector<column_view> const& predicate_columns,
  std::string const& predicate_udf,
  std::vector<column_view> const& filter_columns,
  bool is_ptx,
  std::optional<void*> user_data    = std::nullopt,
  null_aware is_null_aware          = null_aware::NO,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates new table by applying a filter function against every
 * element of the input columns.
 *
 * Null values in the input columns are considered as not matching the filter.
 *
 * Computes:
 * `out[i]... = predicate(columns[i]... ) ? (columns[i]...): not-applied`.
 *
 * @throws std::invalid_argument if the output or any of the inputs are not fixed-width or string
 * types
 * @throws cudf::logic_error if JIT is not supported by the runtime
 *
 * @param predicate_table The table used for predicate expression evaluation
 * @param predicate_expr The predicate filter expression
 * @param filter_table The table to be filtered
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The filtered table
 */
std::unique_ptr<table> filter(
  table_view const& predicate_table,
  ast::expression const& predicate_expr,
  table_view const& filter_table,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace CUDF_EXPORT cudf
