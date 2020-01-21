/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>

#include <memory>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

/**
 * @brief Internal API to calculate groupwise sum
 * 
 * @param values Grouped values to get sum of
 * @param group_sizes Number of valid elements per group
 * @param group_labels ID of group that the corresponding value belongs to
 * @param mr Memory resource to allocate output with
 * @param stream Stream to perform computation in
 */
std::unique_ptr<column> group_sum(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

/**
 * @brief Internal API to calculate groupwise minimum value
 * 
 * @param values Grouped values to get minimum from
 * @param group_sizes Number of valid elements per group
 * @param group_labels ID of group that the corresponding value belongs to
 * @param mr Memory resource to allocate output with
 * @param stream Stream to perform computation in
 */
std::unique_ptr<column> group_min(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

/**
 * @brief Internal API to calculate groupwise maximum value
 * 
 * @param values Grouped values to get maximum from
 * @param group_sizes Number of valid elements per group
 * @param group_labels ID of group that the corresponding value belongs to
 * @param mr Memory resource to allocate output with
 * @param stream Stream to perform computation in
 */
std::unique_ptr<column> group_max(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

/**
 * @brief Internal API to calculate number of non-null values in each group of
 *  @p values
 * 
 * @param values Grouped values to get count of
 * @param group_labels ID of group that the corresponding value belongs to
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param mr Memory resource to allocate output with
 * @param stream Stream to perform computation in
 */
std::unique_ptr<column> group_count(
    column_view const& values,
    rmm::device_vector<size_type> const& group_labels,
    size_type num_groups,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

/**
 * @brief Internal API to calculate groupwise variance
 * 
 * @param values Grouped values to get variance of
 * @param group_means Pre-calculated groupwise MEAN
 * @param group_sizes Number of valid elements per group
 * @param group_labels ID of group corresponding value in @p values belongs to
 * @param ddof Delta degrees of freedom. The divisor used in calculation of 
 *             `var` is `N - ddof`, where `N` is the group size.
 * @param mr Memory resource to allocate output with
 * @param stream Stream to perform computation in
 */
std::unique_ptr<column> group_var(
    column_view const& values,
    column_view const& group_means,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    size_type ddof,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

/**
 * @brief Internal API to calculate groupwise quantiles
 * 
 * @param values Grouped and sorted (within group) values to get quantiles from
 * @param group_sizes Number of valid elements per group
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param quantiles List of quantiles q where q lies in [0,1]
 * @param interp Method to use when desired value lies between data points
 * @param mr Memory resource to allocate output with
 * @param stream Stream to perform computation in
 */
std::unique_ptr<column> group_quantiles(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_offsets,
    std::vector<double> const& quantiles,
    interpolation interp,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
