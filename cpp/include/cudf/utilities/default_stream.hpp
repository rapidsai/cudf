/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup default_stream
 * @{
 * @file
 */

/**
 * @brief Get the current default stream
 *
 * @return The current default stream.
 */
rmm::cuda_stream_view const get_default_stream();

/**
 * @brief Check if per-thread default stream is enabled.
 *
 * @return true if PTDS is enabled, false otherwise.
 */
bool is_ptds_enabled();

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
