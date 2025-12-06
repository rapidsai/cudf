
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <jit/rtc/rtc.hpp>

#include <span>

namespace cudf {
namespace rtc {

header_map get_jit_headers();

header_map get_jit_lto_headers();

std::span<char const*> get_jit_options();

std::span<char const*> get_jit_lto_options();

// input: LTO IR data
// output: nvJitLink object
int32_t get_current_device_physical_model()
{
  int32_t device;
  //   cudaGetDeviceCount()
  CUDF_EXPECTS(cudaGetDevice(&device) == cudaSuccess, "Failed to get current CUDA device");

  cudaDeviceProp props;
  CUDF_EXPECTS(cudaGetDeviceProperties(&props, device) == cudaSuccess,
               "Failed to get device properties");

  return props.major * 10 + props.minor;
}

std::vector<int32_t> get_all_device_physical_models();

}  // namespace rtc
}  // namespace cudf
