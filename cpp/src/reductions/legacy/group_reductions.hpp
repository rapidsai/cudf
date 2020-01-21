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

#include <cudf/cudf.h>
#include <cudf/types.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <vector>

namespace cudf {
namespace detail {

/**
 * @brief Internal API to calculate groupwise standard deviation
 * 
 * @param values Grouped and sorted (within group) values to get quantiles from
 * @param group_labels Vector of size(@p values) indicating the group that each value belongs to
 * @param group_sizes Number of valid elements per group
 * @param result Pre-allocated column to put output of this operation into
 * @param ddof Delta Degrees of Freedom: the divisor used in calculations is `N - ddof`, where `N` is the group size
 * @param stream Stream to perform computation in
 */
void group_std(gdf_column const& values,
               rmm::device_vector<size_type> const& group_labels,
               rmm::device_vector<size_type> const& group_sizes,
               gdf_column * result,
               size_type ddof = 1,
               cudaStream_t stream = 0);

/**
 * @brief Internal API to calculate groupwise variance
 * 
 * @param values Grouped and sorted (within group) values to get quantiles from
 * @param group_labels Vector of size(@p values) indicating the group that each value belongs to
 * @param group_sizes Number of valid elements per group
 * @param result Pre-allocated column to put output of this operation into
 * @param ddof Delta Degrees of Freedom: the divisor used in calculations is `N - ddof`, where `N` is the group size
 * @param stream Stream to perform computation in
 */
void group_var(gdf_column const& values,
               rmm::device_vector<size_type> const& group_labels,
               rmm::device_vector<size_type> const& group_sizes,
               gdf_column * result,
               size_type ddof = 1,
               cudaStream_t stream = 0);

} // namespace detail
} // namespace cudf
