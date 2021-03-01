/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

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
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sum_scan(column_view const& values,
                                 size_type num_groups,
                                 rmm::device_vector<size_type> const& group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise cumulative minimum value
 *
 * @param values Grouped values to get minimum from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> min_scan(column_view const& values,
                                 size_type num_groups,
                                 rmm::device_vector<size_type> const& group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise cumulative maximum value
 *
 * @param values Grouped values to get maximum from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> max_scan(column_view const& values,
                                 size_type num_groups,
                                 rmm::device_vector<size_type> const& group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate cumulative number of values in each group of @p values
 *
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> count_scan(rmm::device_vector<size_type> const& group_offsets,
                                   size_type num_groups,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to collect shifted values within a group
 *
 * If shifted index is out of range of group size, the resulting value will be null.
 *
 * @param values Grouped values to shift
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param num_groups Number of groups
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_shift(column_view const& values,
                                    rmm::device_vector<size_type> const& group_offsets,
                                    size_type num_groups,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
