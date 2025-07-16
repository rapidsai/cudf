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

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <vector>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_sort
 * @{
 * @file
 * @brief Column APIs for sort and rank
 */

/**
 * @brief Computes the row indices that would produce `input` in a lexicographical sorted order.
 *
 * @param input The table to sort
 * @param column_order The desired sort order for each column. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns will be sorted
 * in ascending order.
 * @param null_precedence The desired order of null compared to other elements
 * for each column. Size must be equal to `input.num_columns()` or empty.
 * If empty, all columns will be sorted in `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A non-nullable column of elements containing the permuted row indices of
 * `input` if it were sorted
 */
std::unique_ptr<column> sorted_order(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the row indices that would produce `input` in a stable
 * lexicographical sorted order.
 *
 * The order of equivalent elements is guaranteed to be preserved.
 *
 * @copydoc cudf::sorted_order
 */
std::unique_ptr<column> stable_sorted_order(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Checks whether the rows of a `table` are sorted in a lexicographical
 *        order.
 *
 * @param table             Table whose rows need to be compared for ordering
 * @param column_order      The expected sort order for each column. Size
 *                          must be equal to `in.num_columns()` or empty. If
 *                          empty, it is expected all columns are in
 *                          ascending order.
 * @param null_precedence   The desired order of null compared to other
 *                          elements for each column. Size must be equal to
 *                          `input.num_columns()` or empty. If empty,
 *                          `null_order::BEFORE` is assumed for all columns.
 *
 * @param stream            CUDA stream used for device memory operations and kernel launches
 * @returns                 true if sorted as expected, false if not
 */
bool is_sorted(cudf::table_view const& table,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence,
               rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Performs a lexicographic sort of the rows of a table
 *
 * @param input The table to sort
 * @param column_order The desired order for each column. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `input`. Size must be equal to
 * `input.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return New table containing the desired sorted order of `input`
 */
std::unique_ptr<table> sort(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a stable lexicographic sort of the rows of a table
 *
 * @copydoc cudf::sort
 */
std::unique_ptr<table> stable_sort(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a key-value sort.
 *
 * Creates a new table that reorders the rows of `values` according to the
 * lexicographic ordering of the rows of `keys`.
 *
 * @throws cudf::logic_error if `values.num_rows() != keys.num_rows()`.
 *
 * @param values The table to reorder
 * @param keys The table that determines the ordering
 * @param column_order The desired order for each column in `keys`. Size must be
 * equal to `keys.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `keys`. Size must be equal to
 * `keys.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The reordering of `values` determined by the lexicographic order of
 * the rows of `keys`.
 */
std::unique_ptr<table> sort_by_key(
  table_view const& values,
  table_view const& keys,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a key-value stable sort.
 *
 * @copydoc cudf::sort_by_key
 */
std::unique_ptr<table> stable_sort_by_key(
  table_view const& values,
  table_view const& keys,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the ranks of input column in sorted order.
 *
 * Rank indicate the position of each element in the sorted column and rank
 * value starts from 1.
 *
 * @code{.pseudo}
 * input = { 3, 4, 5, 4, 1, 2}
 * Result for different rank_method are
 * FIRST    = {3, 4, 6, 5, 1, 2}
 * AVERAGE  = {3, 4.5, 6, 4.5, 1, 2}
 * MIN      = {3, 4, 6, 4, 1, 2}
 * MAX      = {3, 5, 6, 5, 1, 2}
 * DENSE    = {3, 4, 5, 4, 1, 2}
 * @endcode
 *
 * @param input The column to rank
 * @param method The ranking method used for tie breaking (same values)
 * @param column_order The desired sort order for ranking
 * @param null_handling  flag to include nulls during ranking. If nulls are not
 * included, corresponding rank will be null.
 * @param null_precedence The desired order of null compared to other elements
 * for column
 * @param percentage flag to convert ranks to percentage in range (0,1]
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column of containing the rank of the each element of the column of `input`. The output
 * column type will be `size_type`column by default or else `double` when
 * `method=rank_method::AVERAGE` or `percentage=True`
 */
std::unique_ptr<column> rank(
  column_view const& input,
  rank_method method,
  order column_order,
  null_policy null_handling,
  null_order null_precedence,
  bool percentage,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns sorted order after sorting each segment in the table.
 *
 * If segment_offsets contains values larger than the number of rows, the behavior is undefined.
 * @throws cudf::logic_error if `segment_offsets` is not `size_type` column.
 *
 * @code{.pseudo}
 * Example:
 * keys = { {9, 8, 7, 6, 5, 4, 3, 2, 1, 0} }
 * offsets = {0, 3, 7, 10}
 * result = cudf::segmented_sorted_order(keys, offsets);
 * result is { 2,1,0, 6,5,4,3, 9,8,7 }
 * @endcode
 *
 * If segment_offsets is empty or contains a single index, no values are sorted
 * and the result is a sequence of integers from 0 to keys.size()-1.
 *
 * The segment_offsets are not required to include all indices. Any indices
 * outside the specified segments will not be sorted.
 *
 * @code{.pseudo}
 * Example: (offsets do not cover all indices)
 * keys = { {9, 8, 7, 6, 5, 4, 3, 2, 1, 0} }
 * offsets = {3, 7}
 * result = cudf::segmented_sorted_order(keys, offsets);
 * result is { 0,1,2, 6,5,4,3, 7,8,9 }
 * @endcode
 *
 * @param keys The table that determines the ordering of elements in each segment
 * @param segment_offsets The column of `size_type` type containing start offset index for each
 * contiguous segment.
 * @param column_order The desired order for each column in `keys`. Size must be
 * equal to `keys.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `keys`. Size must be equal to
 * `keys.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate any returned objects
 * @return sorted order of the segment sorted table
 *
 */
std::unique_ptr<column> segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Returns sorted order after stably sorting each segment in the table.
 *
 * @copydoc cudf::segmented_sorted_order
 */
std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a lexicographic segmented sort of a table
 *
 * If segment_offsets contains values larger than the number of rows, the behavior is undefined.
 * @throws cudf::logic_error if `values.num_rows() != keys.num_rows()`.
 * @throws cudf::logic_error if `segment_offsets` is not `size_type` column.
 *
 * @code{.pseudo}
 * Example:
 * keys = { {9, 8, 7, 6, 5, 4, 3, 2, 1, 0} }
 * values = { {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'} }
 * offsets = {0, 3, 7, 10}
 * result = cudf::segmented_sort_by_key(keys, values, offsets);
 * result is { 'c','b','a', 'g','f','e','d', 'j','i','h' }
 * @endcode
 *
 * If segment_offsets is empty or contains a single index, no values are sorted
 * and the result is a copy of the values.
 *
 * The segment_offsets are not required to include all indices. Any indices
 * outside the specified segments will not be sorted.
 *
 * @code{.pseudo}
 * Example: (offsets do not cover all indices)
 * keys = { {9, 8, 7, 6, 5, 4, 3, 2, 1, 0} }
 * values = { {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'} }
 * offsets = {3, 7}
 * result = cudf::segmented_sort_by_key(keys, values, offsets);
 * result is { 'a','b','c', 'g','f','e','d', 'h','i','j' }
 * @endcode
 *
 * @param values The table to reorder
 * @param keys The table that determines the ordering of elements in each segment
 * @param segment_offsets The column of `size_type` type containing start offset index for each
 * contiguous segment.
 * @param column_order The desired order for each column in `keys`. Size must be
 * equal to `keys.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `keys`. Size must be equal to
 * `keys.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate any returned objects
 * @return table with elements in each segment sorted
 *
 */
std::unique_ptr<table> segmented_sort_by_key(
  table_view const& values,
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a stably lexicographic segmented sort of a table
 *
 * @copydoc cudf::segmented_sort_by_key
 */
std::unique_ptr<table> stable_segmented_sort_by_key(
  table_view const& values,
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the top k values of a column
 *
 * This performs the equivalent of a sort and the slice of the resulting first k elements.
 * However, the returned column may or may not necessarily be sorted.
 *
 * @throw std::invalid_argument if k is greater than the number of rows in the column
 *
 * @param col Column to compute top k
 * @param k Number of values to return
 * @param sort_order The desired sort order for the top k values.
 *                   Default is high to low.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column with the top k values of the input column.
 */
std::unique_ptr<column> top_k(
  column_view const& col,
  size_type k,
  order sort_order                  = order::DESCENDING,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the indices of the top k values of a column
 *
 * The indices will represent the top k elements but may or may not represent
 * those elements as k sorted values.
 *
 * @throw std::invalid_argument if k is greater than the number of rows in the column
 *
 * @param col Column to compute top k
 * @param k Number of values to return
 * @param sort_order The desired sort order for the top k values.
 *                   Default is high to low.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Indices of the top k values of the input column
 */
std::unique_ptr<column> top_k_order(
  column_view const& col,
  size_type k,
  order sort_order                  = order::DESCENDING,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
