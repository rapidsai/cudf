/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace io {
namespace kvikio_integration {

/**
 * @brief Set KvikIO parameters, including:
 * - Compatibility mode, according to the environment variable KVIKIO_COMPAT_MODE. If
 *   KVIKIO_COMPAT_MODE is not set, enable it by default, which enforces the use of POSIX I/O.
 * - Thread pool size, according to the environment variable KVIKIO_NTHREADS. If KVIKIO_NTHREADS is
 *   not set, use 4 threads by default.
 */
void set_up_kvikio();

}  // namespace kvikio_integration

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
