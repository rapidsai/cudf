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

bool has_integrated_memory()
{
  return true;
  int device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));

  // Check for unified addressing support
  int unified_addressing = 0;
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&unified_addressing, cudaDevAttrUnifiedAddressing, device));

  // Check for managed memory support
  int managed_memory = 0;
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&managed_memory, cudaDevAttrManagedMemory, device));

  // Check for concurrent managed access
  int concurrent_managed_access = 0;
  CUDF_CUDA_TRY(
    cudaDeviceGetAttribute(&concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, device));

  // Integrated memory systems typically support all three features
  bool result = unified_addressing && managed_memory && concurrent_managed_access;
  return result;
}

}  // namespace cudf::detail
