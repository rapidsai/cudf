/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
namespace io {
namespace cufile_integration {

/**
 * @brief Returns true if cuFile and its compatibility mode are enabled.
 */
bool is_always_enabled();

/**
 * @brief Returns true if only direct IO through cuFile is enabled (compatibility mode is disabled).
 */
bool is_gds_enabled();

/**
 * @brief Returns true if KvikIO is enabled.
 */
bool is_kvikio_enabled();

/**
 * @brief Set KvikIO parameters, including:
 * - Compatibility mode, according to the environment variable KVIKIO_COMPAT_MODE. If
 *   KVIKIO_COMPAT_MODE is not set, enable it by default, which enforces the use of POSIX I/O.
 * - Thread pool size, according to the environment variable KVIKIO_NTHREADS. If KVIKIO_NTHREADS is
 *   not set, use 4 threads by default.
 */
void set_up_kvikio();

}  // namespace cufile_integration

namespace nvcomp_integration {

/**
 * @brief Returns true if all nvCOMP uses are enabled.
 */
bool is_all_enabled();

/**
 * @brief Returns true if stable nvCOMP use is enabled.
 */
bool is_stable_enabled();

}  // namespace nvcomp_integration
}  // namespace io
}  // namespace CUDF_EXPORT cudf
