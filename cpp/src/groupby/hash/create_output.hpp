/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::groupby::detail::hash {

/**
 * @brief Create the table containing columns for storing aggregation results.
 *
 * @param output_size Number of rows in the output table
 * @param values The values columns to be aggregated
 * @param agg_kinds The aggregation kinds corresponding to each input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The table containing columns for storing aggregation results
 */
std::unique_ptr<table> create_results_table(size_type output_size,
                                            table_view const& values,
                                            host_span<aggregation::Kind const> agg_kinds,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Compute and return an array containing indices of (unique) keys in `key_set`, along
 * with a mapping array from the original input keys to their positions in that array.
 *
 * Note that in the second mapping array, only the keys with indices store in the first array are
 * mapped, leaving other keys with uninitialized mapping values. Since the first array contains
 * indices of all unique keys, this mapping array should cover all input keys.
 *
 * @tparam SetType Type of the key hash set
 *
 * @param key_set Key hash set
 * @param num_keys Number of total keys
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A pair of arrays, the first one contains indices of unique keys retrieved from `key_set`
 *         and the second one maps each of these unique keys from the input keys table to its
 *         corresponding position in first output array.
 */
template <typename SetType>
std::pair<rmm::device_uvector<size_type>, rmm::device_uvector<size_type>> extract_populated_keys(
  SetType const& key_set, size_type num_keys, rmm::cuda_stream_view stream);

/**
 * @brief Compute and return an array mapping each input row to its corresponding key index in
 * the input keys table.
 *
 * @tparam SetType Type of the key hash set
 * @param row_bitmask Bitmask indicating which rows in the input keys table are valid
 * @param key_set Key hash set
 * @param num_rows Number of rows in the input keys table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A device vector mapping each input row to its key index
 */
template <typename SetRef>
rmm::device_uvector<size_type> compute_key_indices(bitmask_type const* row_bitmask,
                                                   SetRef set_ref,
                                                   size_type num_rows,
                                                   rmm::cuda_stream_view stream);

/**
 * @brief Transform (in-place) from the positions of the keys in the input keys table into positions
 * of these keys in the output unique keys table.
 *
 * Note that the positions (indices) of all output unique keys must be covered in the array
 * `key_index_map`. This is guaranteed as it was generated in `extract_populated_keys` function.
 *
 * @param[in,out] key_indices The indices of the keys to transform
 * @param key_index_map The mapping array from the input keys table to the output unique keys table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A device vector mapping each input row to its output row index.
 */
void transform_key_indices(device_span<size_type> key_indices,
                           device_span<size_type const> key_index_map,
                           rmm::cuda_stream_view stream);

/**
 * @brief Enum to indicate whether to compute null counts for aggregation result columns.
 */
enum class compute_null_count { YES, NO };

/**
 * @brief Collect aggregation result columns from a table into a cache object.
 *
 * @param find_null_count Whether to compute null counts for aggregation result columns
 * @param values The values columns
 * @param aggregations The aggregation to compute corresponding to each values column
 * @param agg_results The table containing columns storing aggregation results
 * @param cache The cache object to store the extracted aggregation results
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void collect_output_to_cache(compute_null_count find_null_count,
                             table_view const& values,
                             std::vector<std::unique_ptr<aggregation>> const& aggregations,
                             std::unique_ptr<table>& agg_results,
                             cudf::detail::result_cache* cache,
                             rmm::cuda_stream_view stream);

}  // namespace cudf::groupby::detail::hash
