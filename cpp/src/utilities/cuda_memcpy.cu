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

#include "cudf/detail/utilities/integer_utils.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

namespace cudf::detail {

namespace {

// Simple kernel to copy between device buffers
CUDF_KERNEL void copy_kernel(char const* __restrict__ src, char* __restrict__ dst, size_t n)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx < n) { dst[idx] = src[idx]; }
}

void copy_pinned(void* dst, void const* src, std::size_t size, rmm::cuda_stream_view stream)
{
  if (size == 0) return;

  if (size < get_kernel_pinned_copy_threshold()) {
    const int block_size = 256;
    auto const grid_size = cudf::util::div_rounding_up_safe<size_t>(size, block_size);
    // We are explicitly launching the kernel here instead of calling a thrust function because the
    // thrust function can potentially call cudaMemcpyAsync instead of using a kernel
    copy_kernel<<<grid_size, block_size, 0, stream.value()>>>(
      static_cast<char const*>(src), static_cast<char*>(dst), size);
  } else {
    CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
  }
}

void copy_pageable(void* dst, void const* src, std::size_t size, rmm::cuda_stream_view stream)
{
  if (size == 0) return;

  CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
}

};  // namespace

void cuda_memcpy_async_impl(
  void* dst, void const* src, size_t size, host_memory_kind kind, rmm::cuda_stream_view stream)
{
  if (kind == host_memory_kind::PINNED) {
    copy_pinned(dst, src, size, stream);
  } else if (kind == host_memory_kind::PAGEABLE) {
    copy_pageable(dst, src, size, stream);
  } else {
    CUDF_FAIL("Unsupported host memory kind");
  }
}

}  // namespace cudf::detail
