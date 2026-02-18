/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <atomic>

namespace cudf {

// Forward declaration
enum class join_kind : int32_t;

namespace detail {

/**
 * @brief Implementation of filtered join using a mark-based multiset hash table.
 *
 * This class extends the base filtered_join to implement join operations using
 * multiset semantics, where duplicate keys are allowed in the hash table.
 * This is used when the build table is the **left** table (`set_as_build_table::LEFT`),
 * which may contain duplicate keys.
 *
 * Instead of the traditional two-pass retrieve + sort/dedup approach, this uses
 * a mark-based algorithm: the probe kernel atomically sets the MSB (mark bit) on
 * matching hash table entries via CAS, then a scan kernel collects marked (semi)
 * or unmarked (anti) entries. This provides implicit deduplication and eliminates
 * O(N log N) sort overhead.
 */
class mark_join : public filtered_join {
 private:
  mutable std::atomic<cudf::size_type> _num_marks{0};  ///< Number of marked entries after probe

  /**
   * @brief Performs either a semi or anti join based on the specified kind
   *
   * @param probe The table to probe the hash table with
   * @param kind The kind of join to perform (SEMI or ANTI)
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_anti_join(
    cudf::table_view const& probe,
    join_kind kind,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @brief Core implementation: mark probe + mark scan
   *
   * Performs the mark-based probe of the hash table followed by a scan to
   * collect result indices. The probe kernel walks the hash table for each
   * probe row and atomically marks matching build entries. The scan kernel
   * then iterates the hash table and collects marked (or unmarked for anti)
   * entries into the output buffer using coalesced shared-memory buffered writes.
   *
   * @tparam CGSize CUDA cooperative group size
   * @tparam ProbingScheme Type of probing scheme (mark-aware)
   * @tparam Comparator Type of equality comparator (mark-aware)
   * @param probe The table to probe the hash table with
   * @param preprocessed_probe Preprocessed probe table for row operators
   * @param kind The kind of join to perform
   * @param probing_scheme The probing scheme instance
   * @param comparator The equality comparator instance
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  template <int32_t CGSize, typename ProbingScheme, typename Comparator>
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_probe_and_scan(
    cudf::table_view const& probe,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
    join_kind kind,
    ProbingScheme probing_scheme,
    Comparator comparator,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @brief Clears all mark bits from the hash table entries
   *
   * Must be called before each probe to ensure independent results when the
   * hash table is reused across multiple semi_join/anti_join calls.
   *
   * @param stream CUDA stream on which to perform operations
   */
  void clear_marks(rmm::cuda_stream_view stream);

 public:
  /**
   * @brief Constructor for mark-based filtered join
   *
   * @param build The table to build the hash table from
   * @param compare_nulls How null values should be compared
   * @param load_factor Target load factor for the hash table
   * @param stream CUDA stream on which to perform operations
   */
  mark_join(cudf::table_view const& build,
            cudf::null_equality compare_nulls,
            double load_factor,
            rmm::cuda_stream_view stream);

  /**
   * @brief Implementation of semi join
   *
   * Returns indices of build table rows that have matching keys in the probe table.
   *
   * @param probe The table to probe the hash table with
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) override;

  /**
   * @brief Implementation of anti join
   *
   * Returns indices of build table rows that do not have matching keys in the probe table.
   *
   * @param probe The table to probe the hash table with
   * @param stream CUDA stream on which to perform operations
   * @param mr Memory resource for allocations
   * @return Device vector of indices representing the join result
   */
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) override;
};

}  // namespace detail
}  // namespace cudf
