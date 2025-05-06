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
