/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::detail {

cudf::size_type num_multiprocessors()
{
  int device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  int num_sms = 0;
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  return num_sms;
}

[[nodiscard]] bool has_integrated_memory()
{
  static auto const cached_result = []() {
    int device = 0;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    int is_integrated = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&is_integrated, cudaDevAttrIntegrated, device));
    return is_integrated == 1;
  }();
  return cached_result;
}

}  // namespace cudf::detail
