/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

/**
 * @brief Enum to control whether factorization statistics should be computed
 */
enum class join_statistics : bool { SKIP = false, COMPUTE = true };

namespace detail {
/**
 * @brief Forward declaration for join factorizer implementation
 */
class join_factorizer_impl;
}  // namespace detail

/**
 * @brief Sentinel value for left-side keys not found in right table
 *
 * This constant is exposed primarily for testing purposes.
 * Application code should check for negative values rather than relying on specific sentinel
 * values.
 */
constexpr size_type FACTORIZE_NOT_FOUND = -1;

/**
 * @brief Sentinel value for right-side rows with null keys (when nulls are not equal)
 *
 * This constant is exposed primarily for testing purposes.
 * Application code should check for negative values rather than relying on specific sentinel
 * values.
 */
constexpr size_type FACTORIZE_RIGHT_NULL = -2;

/**
 * @brief Factorizes keys from two tables into unique integer IDs with cardinality metadata
 *
 * This class performs factorization of keys from a right table and optional left table(s)
 * for join operations. Each distinct key in the right table is assigned a unique non-negative
 * integer ID (factor). Rows with equal keys will map to the same ID. Keys that cannot be mapped
 * (e.g., not found in left table, or null keys when nulls are unequal) receive negative sentinel
 * values. The specific ID values are stable for the lifetime of this object but are otherwise
 * unspecified.
 *
 * In addition to key factorization, this class tracks important cardinality metadata:
 * - Distinct count: number of unique keys in the right table
 * - Max duplicate count: maximum frequency of any single key
 *
 * @note The right table must remain valid for the lifetime of this object,
 *       as the hash table references it directly without copying.
 * @note All NaNs are considered equal
 */
class join_factorizer {
 public:
  join_factorizer() = delete;
  ~join_factorizer();
  join_factorizer(join_factorizer const&)            = delete;
  join_factorizer(join_factorizer&&)                 = delete;
  join_factorizer& operator=(join_factorizer const&) = delete;
  join_factorizer& operator=(join_factorizer&&)      = delete;

  /**
   * @brief Constructs a join factorizer from the given right keys.
   *
   * The constructor builds a deduplicating hash table and optionally computes factorization
   * statistics (distinct count and max duplicate count).
   *
   * @throw cudf::logic_error if the right table has no columns
   *
   * @param right The right table containing the keys to factorize
   * @param compare_nulls Controls whether null key values should match or not.
   *        When EQUAL, null keys are treated as equal and assigned a valid non-negative ID.
   *        When UNEQUAL, rows with null keys receive a negative sentinel value.
   * @param statistics Controls whether to compute distinct_count and max_duplicate_count.
   *        If COMPUTE (default), compute statistics for later retrieval via distinct_count()
   *        and max_duplicate_count(). If SKIP, skip statistics computation for better
   *        performance; calling distinct_count() or max_duplicate_count() will throw.
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  join_factorizer(cudf::table_view const& right,
                  null_equality compare_nulls      = null_equality::EQUAL,
                  cudf::join_statistics statistics = cudf::join_statistics::COMPUTE,
                  rmm::cuda_stream_view stream     = cudf::get_default_stream());

  /**
   * @brief Factorize right keys to integer IDs.
   *
   * Computes the factorized right table from the cached right keys. This does not cache
   * the factorized table; each call will recompute it from the internal hash table.
   *
   * For each row in the cached right table, returns the integer ID (factor) assigned to that key.
   * Non-negative integers represent valid factorized keys, while negative values represent
   * keys that cannot be factorized (e.g., null keys when nulls are unequal).
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   *
   * @return A column of INT32 values with the factorized key IDs
   */
  [[nodiscard]] std::unique_ptr<cudf::column> factorize_right_keys(
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Factorize left keys to integer IDs.
   *
   * For each row in the input, returns the integer ID (factor) assigned to that key.
   * Non-negative integers represent keys found in the right table, while negative values
   * represent keys that were not found or cannot be matched (e.g., null keys when nulls
   * are unequal, or keys not present in the right table).
   *
   * @throw std::invalid_argument if keys has different number of columns than right table
   * @throw cudf::data_type_error if keys has different column types than right table
   *
   * @param keys The left keys to factorize (must have same schema as right table)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   *
   * @return A column of INT32 values with the factorized key IDs
   */
  [[nodiscard]] std::unique_ptr<cudf::column> factorize_left_keys(
    cudf::table_view const& keys,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Check if statistics (distinct_count, max_duplicate_count) were computed.
   *
   * @return true if statistics are available, false if statistics was SKIP during construction
   */
  [[nodiscard]] bool has_statistics() const;

  /**
   * @brief Get the number of distinct keys in the right table
   *
   * @throw cudf::logic_error if statistics was SKIP during construction
   *
   * @return The count of unique key combinations in the right table
   */
  [[nodiscard]] size_type distinct_count() const;

  /**
   * @brief Get the maximum number of times any single key appears
   *
   * @throw cudf::logic_error if statistics was SKIP during construction
   *
   * @return The maximum duplicate count across all distinct keys
   */
  [[nodiscard]] size_type max_duplicate_count() const;

 private:
  using impl_type = cudf::detail::join_factorizer_impl;

  std::unique_ptr<impl_type> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
