/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf/column/column.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace groupby {
namespace detail {
/**
 * @brief Internal API to calculate groupwise cumulative sum
 *
 * @param values Grouped values to get sum of
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> sum_scan(column_view const& values,
                                 size_type num_groups,
                                 device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise cumulative minimum value
 *
 * @param values Grouped values to get minimum from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> min_scan(column_view const& values,
                                 size_type num_groups,
                                 device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise cumulative maximum value
 *
 * @param values Grouped values to get maximum from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> max_scan(column_view const& values,
                                 size_type num_groups,
                                 device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate cumulative number of values in each group
 *
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type INT32 of count values
 */
std::unique_ptr<column> count_scan(device_span<size_type const> group_labels,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise min rank value
 *
 * @param grouped_values column or struct column that rows within a group are sorted by
 * @param value_order column of type INT32 that contains the order of the values in the
 * grouped_values column
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets group index offsets with group ID indices
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type size_type of rank values
 */
std::unique_ptr<column> min_rank_scan(column_view const& grouped_values,
                                      column_view const& value_order,
                                      device_span<size_type const> group_labels,
                                      device_span<size_type const> group_offsets,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise max rank value
 *
 * @details  @copydetails min_rank_scan(column_view const& grouped_values,
 *                                      column_view const& value_order,
 *                                      device_span<size_type const> group_labels,
 *                                      device_span<size_type const> group_offsets,
 *                                      rmm::cuda_stream_view stream,
 *                                      rmm::mr::device_memory_resource* mr)
 */
std::unique_ptr<column> max_rank_scan(column_view const& grouped_values,
                                      column_view const& value_order,
                                      device_span<size_type const> group_labels,
                                      device_span<size_type const> group_offsets,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise first rank value
 *
 * @details  @copydetails min_rank_scan(column_view const& grouped_values,
 *                                      column_view const& value_order,
 *                                      device_span<size_type const> group_labels,
 *                                      device_span<size_type const> group_offsets,
 *                                      rmm::cuda_stream_view stream,
 *                                      rmm::mr::device_memory_resource* mr)
 */
std::unique_ptr<column> first_rank_scan(column_view const& grouped_values,
                                        column_view const& value_order,
                                        device_span<size_type const> group_labels,
                                        device_span<size_type const> group_offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise average rank value
 *
 * @details  @copydetails min_rank_scan(column_view const& grouped_values,
 *                                      column_view const& value_order,
 *                                      device_span<size_type const> group_labels,
 *                                      device_span<size_type const> group_offsets,
 *                                      rmm::cuda_stream_view stream,
 *                                      rmm::mr::device_memory_resource* mr)
 */
std::unique_ptr<column> average_rank_scan(column_view const& grouped_values,
                                          column_view const& value_order,
                                          device_span<size_type const> group_labels,
                                          device_span<size_type const> group_offsets,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise dense rank value
 *
 * @param grouped_values column or struct column that rows within a group are sorted by
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets group index offsets with group ID indices
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type size_type of dense rank values
 */
std::unique_ptr<column> dense_rank_scan(column_view const& grouped_values,
                                        column_view const& value_order,
                                        device_span<size_type const> group_labels,
                                        device_span<size_type const> group_offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr);

/**
 * @brief Convert groupwise rank to groupwise percentage rank
 *
 * @param method rank method
 * @param percentage enum to denote the type of conversion ranks to percentage in range (0,1]
 * @param rank Groupwise rank column
 * @param count Groupwise count column
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets group index offsets with group ID indices
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type double of rank values

 */
std::unique_ptr<column> group_rank_to_percentage(rank_method const method,
                                                 rank_percentage const percentage,
                                                 column_view const& rank,
                                                 column_view const& count,
                                                 device_span<size_type const> group_labels,
                                                 device_span<size_type const> group_offsets,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
