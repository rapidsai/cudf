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
   * @brief Construct a sort-merge join object
   *
   * @note The `sort_merge_join` object must not outlive the tables viewed by `left` and `right`,
   * else behavior is undefined.
   *
   * @param left The left table
   * @param is_left_sorted Boolean to indicate if left table is pre-sorted
   * @param right The right table
   * @param is_right_sorted Boolean to indicate if right table is pre-sorted
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the table and columns' device
   * memory.
   */
  sort_merge_join(table_view const& left,
                  bool is_left_sorted,
                  table_view const& right,
                  bool is_right_sorted,
                  null_equality compare_nulls       = null_equality::EQUAL,
                  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables. @see cudf::inner_join().
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing an inner join between two tables
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(rmm::cuda_stream_view stream      = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Helper struct to pre-process the tables before join operations can be performed
   */
  struct preprocessed_table {
    table_view raw_tbl_view;  ///< raw table view before pre processing

    table_view tbl_view;  ///< processed table view which is the null-free subset of the rows of the
                          ///< raw table view if null equality is set to false, and is equal to the
                          ///< raw table view otherwise

    std::optional<rmm::device_buffer> raw_validity_mask =
      std::nullopt;  ///< optional filters for null_equality::UNEQUAL
    std::optional<size_type> raw_num_nulls =
      std::nullopt;  ///< optional filters for null_equality::UNEQUAL
    std::optional<std::unique_ptr<table>> tbl =
      std::nullopt;  ///< optional filters for null_equality::UNEQUAL

    std::optional<std::unique_ptr<column>> tbl_sorted_order =
      std::nullopt;  ///< optional reordering if we are given pre-sorted tables
                    
    /**
     * @brief Mark rows in raw table with nulls at root or child levels by populating the
     * raw_validity_mask
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     * @param mr Device memory resource used to allocate the table and columns' device
     * memory.
     */
    void populate_nonnull_filter(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
    /**
     * @brief Apply raw_validity_mask on the raw_tbl_view to create a null-free table
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     * @param mr Device memory resource used to allocate the table and columns' device
     * memory.
     */
    void apply_nonnull_filter(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
    /**
     * @brief Pre-process the raw table in the case where nulls are unequal.
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     * @param mr Device memory resource used to allocate the table and columns' device
     * memory.
     */
    void preprocess_raw_table(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
    /**
     * @brief Get sorted ordering of processed table
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     * @param mr Device memory resource used to allocate the table and columns' device
     * memory.
     */
    void get_sorted_order(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
    /**
     * @brief Get mapping from processed table to raw table to return correct join indices
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     * @param mr Device memory resource used to allocate the table and columns' device
     * memory.
     * @return A device vector indicating the mapping between pre-processed table and raw table
     */
    rmm::device_uvector<size_type> map_tbl_to_raw(rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr);
  };

 private:
  preprocessed_table ptleft;
  preprocessed_table ptright;
  null_equality compare_nulls;

  /**
   * @brief Preprocess left and right tables before the merge operation
   *
   * @param left The left table
   * @param right The right table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the table and columns' device
   * memory.
   */
  void preprocess_tables(table_view const left,
                         table_view const right,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr);
  /**
   * @brief Post-process left and right tables after the merge operation
   *
   * @param smaller_indices Indices for the smaller processed table used to construct the join
   * result
   * @param larger_indices Indices for the larger processed table used to construct the join result
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the table and columns' device
   * memory.
   * @return A pair of device vectors [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing an inner join between two pre-processed tables
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  postprocess_indices(std::unique_ptr<rmm::device_uvector<size_type>> smaller_indices,
                      std::unique_ptr<rmm::device_uvector<size_type>> larger_indices,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr);
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
 * @brief Assumes pre-sorted inputs and performs only the merge step. Returns a pair of row index
 * vectors corresponding to an inner join between the specified tables.
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
merge_inner_join(cudf::table_view const& left_keys,
                 cudf::table_view const& right_keys,
                 null_equality compare_nulls       = null_equality::EQUAL,
                 rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                 rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
