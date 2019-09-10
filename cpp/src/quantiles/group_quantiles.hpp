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
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <vector>

namespace cudf {
namespace detail {

void group_quantiles(gdf_column const& values,
                     rmm::device_vector<gdf_size_type> const& group_offsets,
                     rmm::device_vector<gdf_size_type> const& group_sizes,
                     gdf_column * result,
                     std::vector<double> const& quantiles,
                     cudf::interpolation interpolation,
                     cudaStream_t stream = 0);

} // namespace detail
} // namespace cudf
