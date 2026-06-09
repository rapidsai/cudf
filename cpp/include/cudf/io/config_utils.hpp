/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cstddef>

namespace CUDF_EXPORT cudf {
namespace io {
//! `KvikIO`
namespace kvikio_integration {

/**
 * @addtogroup io_configuration
 * @{
 * @file
 */

/**
 * @brief Set KvikIO parameters
 *
 * Parameters include:
 * - Compatibility mode, according to the environment variable KVIKIO_COMPAT_MODE. If
 *   KVIKIO_COMPAT_MODE is not set, enable it by default, which enforces the use of POSIX I/O.
 * - Thread pool size, according to the environment variable KVIKIO_NTHREADS. If KVIKIO_NTHREADS is
 *   not set, use 4 threads by default.
 */
void set_up_kvikio();

}  // namespace kvikio_integration

//! `nvCOMP`
namespace nvcomp_integration {

/**
 * @brief Returns true if all nvCOMP uses are enabled
 *
 * @return true if all nvCOMP uses are enabled
 */
[[nodiscard]] bool is_all_enabled();

/**
 * @brief Returns true if stable nvCOMP use is enabled
 *
 * @return true true if stable nvCOMP use is enabled
 */
[[nodiscard]] bool is_stable_enabled();

}  // namespace nvcomp_integration

//! IO Integrated Memory Optimization
namespace integrated_memory_optimization {

/**
 * @brief Returns true if integrated memory optimizations are enabled
 *
 * Controlled by the LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION environment variable.
 * Valid values: "AUTO" (default), "ON", "OFF"
 * - AUTO: Use hardware detection (cudaDevAttrIntegrated)
 * - ON: Always enable optimization
 * - OFF: Always disable optimization
 *
 * @return true if integrated memory optimizations are enabled
 */
[[nodiscard]] bool is_enabled();

/** @} */  // end of group
}  // namespace integrated_memory_optimization

//! Parquet
namespace parquet_reader {

/**
 * @brief Returns the Parquet reader's footer speculative read size in bytes.
 *
 * Controlled by the `LIBCUDF_PARQUET_METADATA_SIZE_HINT` environment variable.
 * Defaults to 64 KiB.
 *
 * When the footer is smaller than the speculative read size, the footer metadata
 * is loaded in a single read, which is especially useful for high-latency, remote
 * storage systems. When the footer is larger than the speculative read size, the
 * footer metadata will be loaded in two reads.
 *
 * Set `LIBCUDF_PARQUET_METADATA_SIZE_HINT=0` to disable speculative reads.
 *
 * @return Number of bytes to speculatively read from the end of the source.
 */
[[nodiscard]] std::size_t metadata_size_hint();

}  // namespace parquet_reader
}  // namespace io
}  // namespace CUDF_EXPORT cudf
