/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/replace.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

#include <memory>

namespace cudf {
namespace groupby {
namespace detail {

/**
 * @brief Internal API to replace nulls with preceding/following non-null values in @p value
 *
 * @param[in] value A column whose null values will be replaced.
 * @param[in] group_labels ID of group that the corresponding value belongs to
 * @param[in] replace_policy Specify the position of replacement values relative to null values.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate device memory of the returned column.
 */
std::unique_ptr<column> group_replace_nulls(cudf::column_view const& value,
                                            rmm::device_vector<cudf::size_type> const& group_labels,
                                            cudf::replace_policy replace_policy,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
