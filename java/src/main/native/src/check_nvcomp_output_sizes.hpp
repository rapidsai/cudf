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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace java {

/**
 * Check that the vector of expected uncompressed sizes matches the vector of actual compressed
 * sizes. Both vectors are assumed to be in device memory and contain num_chunks elements.
 */
bool check_nvcomp_output_sizes(std::size_t const* dev_uncompressed_sizes,
                               std::size_t const* dev_actual_uncompressed_sizes,
                               std::size_t num_chunks,
                               rmm::cuda_stream_view stream);
}  // namespace java
}  // namespace cudf
