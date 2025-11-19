/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <optional>
#include <variant>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

/**
 * @brief Class that implements sort-merge algorithm for table joins
 */
class sort_merge_join {
 public:
  sort_merge_join()                                  = delete;
  sort_merge_join(sort_merge_join const&)            = delete;
  sort_merge_join(sort_merge_join&&)                 = delete;
  sort_merge_join& operator=(sort_merge_join const&) = delete;
  sort_merge_join& operator=(sort_merge_join&&)      = delete;

  /**
   * @brief Construct a sort-merge join object that pre-processes the right table
   * on creation, and can be used on subsequent join operations with multiple
   * left tables.
   *
   * @note The `sort_merge_join` object must not outlive the table viewed by `right`,
   * else behavior is undefined.
   *
   * @param right The right table
   * @param is_right_sorted Enum to indicate if right table is pre-sorted
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  sort_merge_join(table_view const& right,
                  sorted is_right_sorted,
                  null_equality compare_nulls  = null_equality::EQUAL,
                  rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * an inner join between the right table passed while creating the sort_merge_join object, and the
   * left table.
   * @see cudf::inner_join().
   *
   * @param left The left table
   * @param is_left_sorted Enum to indicate if left table is pre-sorted
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device
   * memory.
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing an inner join between two tables
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(table_view const& left,
             sorted is_left_sorted,
             rmm::cuda_stream_view stream      = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Returns context information about matches between the left and right tables.
   *
   * This method computes, for each row in the left table, how many matching rows exist in
   * the right table according to inner join semantics, and returns the number of matches through a
   * match_context object.
   *
   * This is particularly useful for:
   * - Determining the total size of a potential join result without materializing it
   * - Planning partitioned join operations for large datasets
   *
   * The returned join_match_context can be used directly with partitioned_inner_join() to
   * process large joins in manageable chunks.
   *
   * @param left The left table to join with the pre-processed right table
   * @param is_left_sorted Enum to indicate if left table is pre-sorted
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A join_match_context object containing the left table view and a device vector
   *         of match counts for each row in the left table
   */
  cudf::join_match_context inner_join_match_context(
    table_view const& left,
    sorted is_left_sorted,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Performs an inner join between a partition of the left table and the right table.
   *
   * This method executes an inner join operation between a specific partition of the left table
   * (defined by the join_partition_context) and the right table that was provided when constructing
   * the sort_merge_join object. The join_partition_context must have been previously created by
   * calling inner_join_match_context().
   *
   * This partitioning approach enables processing large joins in smaller, memory-efficient chunks,
   * while maintaining consistent results as if the entire join was performed at once. This is
   * particularly useful for handling large datasets that would otherwise exceed available memory
   * resources.
   *
   * The returned indices can be used to construct the join result for this partition. The
   * left_indices are relative to the original complete left table (not just the partition), so they
   * can be used directly with the original left table to extract matching rows.
   *
   * @param context The partition context containing match information and partition bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] containing the row indices
   *         from both tables that satisfy the join condition for this partition. The left_indices
   *         are relative to the complete left table, not just the partition.
   *
   * @code{.cpp}
   * // Create join object with pre-processed right table
   * sort_merge_join join_obj(right_table, sorted::NO);
   *
   * // Get match context for the entire left table
   * auto context = join_obj.inner_join_match_context(left_table, sorted::NO);
   *
   * // Define partition boundaries (e.g., process 1000 rows at a time)
   * for (size_type start = 0; start < left_table.num_rows(); start += 1000) {
   *   size_type end = std::min(start + 1000, left_table.num_rows());
   *
   *   // Create partition context
   *   cudf::join_partition_context part_ctx{context, start, end};
   *
   *   // Get join indices for this partition
   *   auto [left_indices, right_indices] = join_obj.partitioned_inner_join(part_ctx);
   *
   *   // Process the partition result...
   * }
   * @endcode
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(
    cudf::join_partition_context const& context,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

 private:
  /**
   * @brief Helper struct to pre-process tables before join operations
   */
  struct preprocessed_table {
    table_view _table_view;  ///< Unprocessed table view before pre-processing

    table_view
      _null_processed_table_view;  ///< Processed table view which is the null-free subset of the
                                   ///< rows of the unprocessed table view if null equality is set
                                   ///< to false, otherwise equal to the unprocessed table view

    std::optional<rmm::device_buffer> _validity_mask =
      std::nullopt;  ///< Optional validity mask for null_equality::UNEQUAL case
    std::optional<size_type> _num_nulls =
      std::nullopt;  ///< Optional count of nulls for null_equality::UNEQUAL case
    std::optional<std::unique_ptr<table>> _null_processed_table =
      std::nullopt;  ///< Optional filtered table for null_equality::UNEQUAL case

    std::optional<std::unique_ptr<column>> _null_processed_table_sorted_order =
      std::nullopt;  ///< Optional sort ordering for pre-sorted tables

    /**
     * @brief Mark rows in unprocessed table with nulls at root or child levels by populating the
     * _validity_mask
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     */
    void populate_nonnull_filter(rmm::cuda_stream_view stream);

    /**
     * @brief Apply _validity_mask to the _table_view to create a null-free table
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     */
    void apply_nonnull_filter(rmm::cuda_stream_view stream);

    /**
     * @brief Pre-process the unprocessed table when null equality is set to unequal
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     */
    void preprocess_unprocessed_table(rmm::cuda_stream_view stream);

    /**
     * @brief Compute sorted ordering of the processed table
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     */
    void get_sorted_order(rmm::cuda_stream_view stream);

    /**
     * @brief Create mapping from processed table indices to unprocessed table indices
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     * @return A device vector containing the mapping from processed table indices to unprocessed
     * table indices
     */
    rmm::device_uvector<size_type> map_table_to_unprocessed(rmm::cuda_stream_view stream);
  };
  preprocessed_table preprocessed_left;
  preprocessed_table preprocessed_right;
  null_equality compare_nulls;

  /**
   * @brief Post-process left and right tables after the merge operation
   *
   * @param smaller_indices Indices for the smaller processed table used to construct the join
   * result
   * @param larger_indices Indices for the larger processed table used to construct the join result
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void postprocess_indices(device_span<size_type> smaller_indices,
                           device_span<size_type> larger_indices,
                           rmm::cuda_stream_view stream);

  /**
   * @brief Core merge operation implementation for the sort-merge join algorithm.
   *
   * This template method performs the actual merge operation between preprocessed left and
   * right tables for different types of joins. It serves as the common implementation for
   * various join methods (inner_join, inner_join_match_context, partitioned_inner_join).
   *
   * The method takes a generic merge operation functor that defines the specific join
   * behavior to be applied during the merge phase. This design allows for different join
   * operations (such as generating indices or counting matches) to share the same
   * underlying merge algorithm.
   *
   * The method expects that tables have already been preprocessed to handle
   * null values according to the null_equality setting.
   *
   * @tparam MergeOperation Functor type that implements the specific join operation
   * @param right_view The preprocessed right table view
   * @param left_view The preprocessed left table view
   * @param op The merge operation functor to execute during the merge
   *
   * @return The result of the merge operation as defined by the MergeOperation functor
   *         (typically pairs of join indices or match counts)
   */
  template <typename MergeOperation>
  auto invoke_merge(table_view right_view, table_view left_view, MergeOperation&& op);
};

/**
 * @brief Returns a pair of row index vectors corresponding to an
 * inner join between the specified tables.
 *
 * The first returned vector contains the row indices from the left
 * table that have a match in the right table (in unspecified order).
 * The corresponding values in the second returned vector are
 * the matched row indices from the right table.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{1, 2}, {0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw std::invalid_argument if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing an inner join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_inner_join(cudf::table_view const& left_keys,
                      cudf::table_view const& right_keys,
                      null_equality compare_nulls       = null_equality::EQUAL,
                      rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to an inner join between the specified
 * tables.
 *
 * Assumes pre-sorted inputs and performs only the merge step.
 * The first returned vector contains the row indices from the left
 * table that have a match in the right table (in unspecified order).
 * The corresponding values in the second returned vector are
 * the matched row indices from the right table.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{1, 2}, {0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw std::invalid_argument if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing an inner join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge_inner_join(cudf::table_view const& left_keys,
                 cudf::table_view const& right_keys,
                 null_equality compare_nulls       = null_equality::EQUAL,
                 rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                 rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
