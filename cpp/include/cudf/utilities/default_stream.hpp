/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

namespace cudf {

/**
 * @brief Default stream for cudf
 *
 * Use this value to ensure the correct stream is used when compiled with per
 * thread default stream.
 */
#if defined(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
static const rmm::cuda_stream_view default_stream_value{rmm::cuda_stream_per_thread};
#else
static constexpr rmm::cuda_stream_view default_stream_value{};
#endif

/**
 * @brief Check if per-thread default stream is enabled.
 *
 * @return true if PTDS is enabled, false otherwise.
 */
bool is_ptds_enabled();

}  // namespace cudf
