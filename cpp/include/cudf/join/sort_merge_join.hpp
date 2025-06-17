/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
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
   /**
    * @brief Holds context information about matches between tables during a join operation.
    * 
    * This structure stores the left table view and a device vector containing the count of 
    * matching rows in the right table for each row in the left table. Used primarily by 
    * inner_join_match_context() to track join match information.
    */
   struct match_context {
     table_view _left_table; // View of the left table involved in the join operation 
     std::unique_ptr<rmm::device_uvector<size_type>> _match_counts; // A device vector containing the count of matching rows in the right table for each row in left table
   };

   /**
    * @brief Stores context information for partitioned join operations.
    * 
    * This structure maintains context for partitioned join operations, containing the match
    * context from a previous join operation along with the start and end indices that define
    * the current partition of the left table being processed.
    * 
    * Used with partitioned_inner_join() to perform large joins in smaller chunks while 
    * preserving the context from the initial match operation.
    */
   struct partition_context {
     match_context left_table_context; // The match context from a previous inner_join_match_context call
     size_type left_start_idx; // The starting row index of the current left table partition
     size_type left_end_idx; // The ending row index (exclusive) of the current left table partition
   };

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
   * the right table according to inner join semantics, and returns the number of matches through a match_context object.
   *
   * This is particularly useful for:
   * - Determining the total size of a potential join result without materializing it
   * - Planning partitioned join operations for large datasets
   *
   * The returned match_context can be used directly with partitioned_inner_join() to 
   * process large joins in manageable chunks.
   *
   * @param left The left table to join with the pre-processed right table
   * @param is_left_sorted Enum to indicate if left table is pre-sorted
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A match_context object containing the left table view and a device vector 
   *         of match counts for each row in the left table
   */
  match_context inner_join_match_context(
    table_view const& left,
    sorted is_left_sorted,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * a partitioned inner join between a specific range of rows from the left table and the
   * right table.
   *
   * This function should be called after `inner_join_size_per_row` to perform a partitioned
   * join operation. The caller typically first uses `inner_join_size_per_row` to determine
   * the join size for each row in the left table, then partitions the left table based on
   * resource constraints, and finally calls this function for each partition.
   *
   * This function assumes that the left table has already been processed by a previous
   * call to `inner_join_size_per_row` using the same `sort_merge_join` object.
   *
   * Example:
   * ```
   * // After determining join sizes
   * auto join_sizes = join_obj.inner_join_size_per_row(left_table, sorted::NO);
   *
   * // Determine partitions based on resource constraints
   * std::vector<std::pair<size_t, size_t>> partitions = compute_partitions(*join_sizes);
   *
   * // Process each partition
   * for (auto [start, end] : partitions) {
   *   auto [left_indices, right_indices] = join_obj.partitioned_inner_join(start, end);
   *   // Use left_indices and right_indices to construct join result for this partition
   * }
   * ```
   * @param left_partition_begin The starting row index of the current left table partition
   * @param left_partition_end The ending row index (exclusive) of the current left table partition
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing an inner join between the right table and the specified
   * partition of the left table. The left_indices will be relative to the overall left table,
   * not just the partition.
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(
    partition_context const &context,
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
    
    template<typename SortedOrderFunc, typename PresortedFunc>
    auto get_iterator(SortedOrderFunc &&sorted_order_iterator, PresortedFunc &&presorted_iterator) -> std::variant<std::decay_t<decltype(sorted_order_iterator())>, std::decay_t<decltype(presorted_iterator())>> {
      if (_null_processed_table_sorted_order.has_value()) {
        return sorted_order_iterator();
      } else {
        return presorted_iterator();
      }
    }

    auto begin() {
      return get_iterator([this]() { return _null_processed_table_sorted_order.value()->view().begin<size_type>(); }, [this](){ return thrust::counting_iterator(0); });
    }

    auto end() {
      return get_iterator([this]() { return _null_processed_table_sorted_order.value()->view().end<size_type>(); }, [this](){ return thrust::counting_iterator(_null_processed_table_view.num_rows()); });
    }

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
  auto invoke_merge(table_view right_view, table_view left_view, MergeOperation &&op);
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
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
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
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
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
