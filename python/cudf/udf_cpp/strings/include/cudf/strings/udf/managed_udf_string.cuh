/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/strings/udf/udf_string.hpp>

namespace cudf::strings::udf {

/**
 * @brief Container for a udf_string and its NRT memory information
 *
 * `meminfo` is a MemInfo struct from numba-cuda, see:
 * https://github.com/NVIDIA/numba-cuda/blob/main/numba_cuda/numba/cuda/memory_management/nrt.cuh
 */
struct managed_udf_string {
  void* meminfo;
  cudf::strings::udf::udf_string udf_str;
};

}  // namespace cudf::strings::udf
