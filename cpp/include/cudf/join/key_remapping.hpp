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

namespace detail {
/**
 * @brief Forward declaration for key remapping implementation
 */
class key_remapping_impl;
}  // namespace detail

/// Sentinel value for probe-side keys not found in build table
constexpr size_type KEY_REMAP_NOT_FOUND = -1;

/// Sentinel value for build-side rows with null keys (when nulls are not equal)
constexpr size_type KEY_REMAP_BUILD_NULL = -2;

/**
 * @brief Remaps keys to unique integer IDs
 *
 * Each distinct key in the build table is assigned a unique non-negative integer ID.
 * Rows with equal keys will map to the same ID. The specific ID values are stable
 * for the lifetime of this object but are otherwise unspecified.
 *
 * @note The build table must remain valid for the lifetime of this object,
 *       as the hash table references it directly without copying.
 * @note All NaNs are considered equal
 */
class key_remapping {
 public:
  key_remapping() = delete;
  ~key_remapping();
  key_remapping(key_remapping const&)            = delete;
  key_remapping(key_remapping&&)                 = delete;
  key_remapping& operator=(key_remapping const&) = delete;
  key_remapping& operator=(key_remapping&&)      = delete;

  /**
   * @brief Constructs a key remapping structure from the given build keys.
   *
   * @throw cudf::logic_error if the build table has no columns
   *
   * @param build The build table containing the keys to remap
   * @param compare_nulls Controls whether null key values should match or not.
   *        When EQUAL, null keys are treated as equal and assigned a valid non-negative ID.
   *        When UNEQUAL, rows with null keys map to KEY_REMAP_BUILD_NULL.
   * @param compute_metrics If true (default), compute distinct_count and max_duplicate_count.
   *        If false, skip metrics computation for better performance; calling get_distinct_count()
   *        or get_max_duplicate_count() will throw.
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  key_remapping(cudf::table_view const& build,
                null_equality compare_nulls  = null_equality::EQUAL,
                bool compute_metrics         = true,
                rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Remap build keys to integer IDs.
   *
   * Recomputes the remapped build table from the cached build keys. This does not cache
   * the remapped table; each call will recompute it from the key remapping.
   *
   * For each row in the cached build table, returns the integer ID assigned to that key.
   * - Keys that match a build table key: return a non-negative integer
   * - Keys with nulls (when compare_nulls is EQUAL): return the ID assigned to null keys
   * - Keys with nulls (when compare_nulls is UNEQUAL): return KEY_REMAP_BUILD_NULL
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   *
   * @return A column of INT32 values with the remapped key IDs
   */
  [[nodiscard]] std::unique_ptr<cudf::column> remap_build_keys(
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Remap probe keys to integer IDs.
   *
   * For each row in the input, returns the integer ID assigned to that key.
   * - Keys that match a build table key: return a non-negative integer
   * - Keys not found in build table: return KEY_REMAP_NOT_FOUND
   * - Keys with nulls (when compare_nulls is EQUAL): return the ID assigned to null keys,
   *   or KEY_REMAP_NOT_FOUND if no null keys exist in build table
   * - Keys with nulls (when compare_nulls is UNEQUAL): return KEY_REMAP_NOT_FOUND
   *
   * @throw std::invalid_argument if keys has different number of columns than build table
   * @throw cudf::data_type_error if keys has different column types than build table
   *
   * @param keys The probe keys to remap (must have same schema as build table)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   *
   * @return A column of INT32 values with the remapped key IDs
   */
  [[nodiscard]] std::unique_ptr<cudf::column> remap_probe_keys(
    cudf::table_view const& keys,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Check if metrics (distinct_count, max_duplicate_count) were computed.
   *
   * @return true if metrics are available, false if compute_metrics was false during construction
   */
  [[nodiscard]] bool has_metrics() const;

  /**
   * @brief Get the number of distinct keys in the build table
   *
   * @throw cudf::logic_error if compute_metrics was false during construction
   *
   * @return The count of unique key combinations found during build
   */
  [[nodiscard]] size_type get_distinct_count() const;

  /**
   * @brief Get the maximum number of times any single key appears
   *
   * @throw cudf::logic_error if compute_metrics was false during construction
   *
   * @return The maximum duplicate count across all distinct keys
   */
  [[nodiscard]] size_type get_max_duplicate_count() const;

 private:
  using impl_type = cudf::detail::key_remapping_impl;

  std::unique_ptr<impl_type> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
