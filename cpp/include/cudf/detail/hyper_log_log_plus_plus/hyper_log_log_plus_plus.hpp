/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace groupby::detail {

/**
 * Compute the hashs of the input column, then generate a scalar that is a sketch in long array
 * format
 */
std::unique_ptr<scalar> reduce_hyper_log_log_plus_plus(column_view const& input,
                                                       int64_t const precision,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr);

/**
 * Merge sketches in long array format, and compute the estimated distinct value(long)
 * Input is a struct column with multiple long columns which is consistent with Spark.
 */
std::unique_ptr<scalar> reduce_merge_hyper_log_log_plus_plus(column_view const& input,
                                                             int64_t const precision,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr);

}  // namespace groupby::detail
}  // namespace cudf
