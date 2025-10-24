/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <utility>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

namespace detail {
/**
 * @brief Forward declaration for our filtered hash join
 */
class filtered_join;
}  // namespace detail

/**
 * @brief Specifies which table to use as the build table in a hash join operation
 * @see filtered_join
 */
enum class set_as_build_table { LEFT, RIGHT };

/**
 * @brief Filtered hash join that builds hash table on creation and probes results in subsequent
 * `*_join` member functions
 *
 * This class enables the filtered hash join scheme that builds hash table once, and probes as many
 * times as needed (possibly in parallel). When the hash table is created from the right table i.e.
 * the table that acts as the filter to be applied on left tables in subsequent `_join` operations,
 * the `cuco::static_set` data structure is used. On the other hand, when the left table is to be
 * reused, the underlying hash table data structure is the `cuco::static_multiset`. Since multiset
 * operations are computationally more expensive that set operations, right table reuse should be
 * preferred if possible.
 *
 * @note All NaNs are considered as equal
 */
class filtered_join {
 public:
  filtered_join() = delete;
  ~filtered_join();
  filtered_join(filtered_join const&)            = delete;
  filtered_join(filtered_join&&)                 = delete;
  filtered_join& operator=(filtered_join const&) = delete;
  filtered_join& operator=(filtered_join&&)      = delete;

  /**
   * @brief Constructs a filtered hash join object for subsequent probe calls
   *
   * @param build The build table
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param reuse_tbl Specifies which table to use as the build table. If LEFT, the build table
   * is considered as the left table and is reused with multiple right (probe) tables. If RIGHT,
   * the build table is considered as the right/filter table and will be applied to multiple left
   * (probe) tables.
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  filtered_join(cudf::table_view const& build,
                cudf::null_equality compare_nulls = null_equality::EQUAL,
                set_as_build_table reuse_tbl      = set_as_build_table::RIGHT,
                rmm::cuda_stream_view stream      = cudf::get_default_stream());

  /**
   * @brief Constructs a filtered hash join object for subsequent probe calls
   *
   * @param build The build table
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param reuse_tbl Specifies which table to use as the build table. If LEFT, the build table
   * is considered as the left table and is reused with multiple right (probe) tables. If RIGHT,
   * the build table is considered as the right/filter table and will be applied to multiple left
   * (probe) tables.
   * @param load_factor The desired ratio of filled slots to total slots in the hash table, must be
   * in range (0,1]. For example, 0.5 indicates a target of 50% occupancy. Note that the actual
   * occupancy achieved may be slightly lower than the specified value.
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  filtered_join(cudf::table_view const& build,
                null_equality compare_nulls  = null_equality::EQUAL,
                set_as_build_table reuse_tbl = set_as_build_table::RIGHT,
                double load_factor           = 0.5,
                rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Returns a vector of row indices corresponding to a semi-join
   * between the specified tables.
   *
   * The returned vector contains the row indices from the left table
   * for which there is a matching row in the right table. Note that the left table
   * is the build table if `reuse_left_table` is set to true, and is the probe table
   * otherwise.
   *
   * @code{.pseudo}
   * TableA: {{0, 1, 2}}
   * TableB: {{1, 2, 3}}
   * Result: {1, 2}
   * @endcode
   *
   * @param probe The probe table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   *
   * @return A vector `left_indices` that can be used to construct
   * the result of performing a left semi join
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> semi_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns a vector of row indices corresponding to a anti-join
   * between the specified tables.
   *
   * The returned vector contains the row indices from the left table
   * for which there are no matching rows in the right table. Note that the left table
   * is the build table if `reuse_left_table` is set to true, and is the probe table
   * otherwise.
   *
   * @code{.pseudo}
   * TableA: {{0, 1, 2}}
   * TableB: {{1, 2, 3}}
   * Result: {1, 2}
   * @endcode
   *
   * @param probe The probe table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   *
   * @return A vector `left_indices` that can be used to construct
   * the result of performing a left anti join
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> anti_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  set_as_build_table _reuse_tbl;
  std::unique_ptr<cudf::detail::filtered_join> _impl;  ///< Filtered hash join implementation
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
