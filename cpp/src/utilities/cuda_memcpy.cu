/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf/detail/utilities/integer_utils.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

#include <vector>

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
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(dst, src, size, cudaMemcpyDefault, stream));
  }
}

void copy_pageable(void* dst, void const* src, std::size_t size, rmm::cuda_stream_view stream)
{
  if (size == 0) return;

  CUDF_CUDA_TRY(cudf::detail::memcpy_async(dst, src, size, cudaMemcpyDefault, stream));
}

bool is_memcpy_batch_async_supported()
{
#if CUDART_VERSION >= 13000
  // cudaMemcpyBatchAsync is supported on all CUDA 13 versions
  return true;
#else
  // For CUDA 12, we check for CUDA runtime >=12.8 and cache the result
  static auto supports_memcpy_batch_async{[] {
    // CUDA 12.8 or higher is required for cudaMemcpyBatchAsync
    int cuda_runtime_version{};
    auto runtime_result = cudaRuntimeGetVersion(&cuda_runtime_version);
    return runtime_result == cudaSuccess and cuda_runtime_version >= 12080;
  }()};
  return supports_memcpy_batch_async;
#endif
}

};  // namespace

cudaError_t memcpy_batch_async(
  void** dsts, void** srcs, std::size_t* sizes, std::size_t count, rmm::cuda_stream_view stream)
{
  // Filter out invalid copies (nullptr dst/src or size==0);
  // cudaMemcpyBatchAsync does not support these inputs
  std::size_t valid_count = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (dsts[i] != nullptr && srcs[i] != nullptr && sizes[i] != 0) { ++valid_count; }
  }
  if (valid_count == 0) { return cudaSuccess; }

  // Build filtered arrays if any copies were invalid
  std::vector<void*> valid_dsts;
  std::vector<void*> valid_srcs;
  std::vector<std::size_t> valid_sizes;
  if (valid_count < count) {
    valid_dsts.reserve(valid_count);
    valid_srcs.reserve(valid_count);
    valid_sizes.reserve(valid_count);
    for (std::size_t i = 0; i < count; ++i) {
      if (dsts[i] != nullptr && srcs[i] != nullptr && sizes[i] != 0) {
        valid_dsts.push_back(dsts[i]);
        valid_srcs.push_back(srcs[i]);
        valid_sizes.push_back(sizes[i]);
      }
    }
    dsts  = valid_dsts.data();
    srcs  = valid_srcs.data();
    sizes = valid_sizes.data();
    count = valid_count;
  }

  // The cudaMemcpyBatchAsync API requires CUDA >= 12.8 and does not support the default stream
  if (is_memcpy_batch_async_supported() && !stream.is_default()) {
#if CUDART_VERSION >= 12080
    cudaMemcpyAttributes attrs[1] = {};  // zero-initialize all fields
    attrs[0].srcAccessOrder       = cudaMemcpySrcAccessOrderStream;
    attrs[0].flags                = cudaMemcpyFlagPreferOverlapWithCompute;
    std::size_t attrs_idxs[1]     = {0};
    std::size_t num_attrs{1};
#if CUDART_VERSION >= 13000
    return cudaMemcpyBatchAsync(
      dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs, stream.value());
#else
    std::size_t fail_idx;
    return cudaMemcpyBatchAsync(
      dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs, &fail_idx, stream.value());
#endif  // CUDART_VERSION >= 13000
#endif  // CUDART_VERSION >= 12080
  }
  // Implement a compatible API for CUDA < 12.8
  for (std::size_t i = 0; i < count; ++i) {
    cudaError_t status =
      cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream.value());
    if (status != cudaSuccess) { return status; }
  }
  return cudaSuccess;
}

cudaError_t memcpy_async(
  void* dst, void const* src, size_t count, cudaMemcpyKind kind, rmm::cuda_stream_view stream)
{
  if (count == 0) { return cudaSuccess; }

  // Use batch API with size 1 to prefer cudaMemcpyBatchAsync over
  // cudaMemcpyAsync. The batched API can be more efficient.
  void* dsts[1]   = {dst};
  void* srcs[1]   = {const_cast<void*>(src)};
  size_t sizes[1] = {count};
  return memcpy_batch_async(dsts, srcs, sizes, 1, stream);
}

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
