/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

namespace CUDF_EXPORT cudf {
//! KvikIO manager
namespace io {

namespace detail {

//! @cond Internal

/**
 * @brief Set the number of IO threads used by KvikIO.
 *
 * If the new value differs from the current one, the following happens in sequence:
 *
 * - The calling thread is blocked until all pending I/O tasks complete.
 * - The old thread pool is destroyed.
 * - A new pool is created.
 *
 * Otherwise, the existing thread pool will be used for subsequent I/O operations.
 *
 * @param num_io_threads The number of IO threads to be used.
 */
void set_num_io_threads(unsigned int num_io_threads);

/**
 * @brief Get the number of IO threads used by KvikIO.
 *
 * @return The number of IO threads used by KvikIO.
 */
[[nodiscard]] unsigned int num_io_threads();

//! @endcond

}  // namespace detail
}  // namespace io
}  // namespace CUDF_EXPORT cudf
