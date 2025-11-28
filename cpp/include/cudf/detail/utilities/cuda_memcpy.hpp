/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

enum class host_memory_kind : uint8_t { PINNED, PAGEABLE };

void cuda_memcpy_async_impl(
  void* dst, void const* src, size_t size, host_memory_kind kind, rmm::cuda_stream_view stream);

/**
 * @brief Wrapper around cudaMemcpyAsync
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param count Size in bytes to copy
 * @param kind Type of memory copy
 * @param stream CUDA stream
 * @return cudaError_t CUDA error code
 */
inline cudaError_t memcpy_async(
  void* dst, void const* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
{
  return cudaMemcpyAsync(dst, src, count, kind, stream);
}

/**
 * @brief Asynchronously copies data from host to device memory.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination device memory
 * @param src Source host memory
 * @param stream CUDA stream used for the copy
 */
template <typename T>
void cuda_memcpy_async(device_span<T> dst, host_span<T const> src, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(dst.size() == src.size(), "Mismatched sizes in cuda_memcpy_async");
  auto const is_pinned = src.is_device_accessible();
  cuda_memcpy_async_impl(dst.data(),
                         src.data(),
                         src.size_bytes(),
                         is_pinned ? host_memory_kind::PINNED : host_memory_kind::PAGEABLE,
                         stream);
}

/**
 * @brief Asynchronously copies data from device to host memory.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination host memory
 * @param src Source device memory
 * @param stream CUDA stream used for the copy
 */
template <typename T>
void cuda_memcpy_async(host_span<T> dst, device_span<T const> src, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(dst.size() == src.size(), "Mismatched sizes in cuda_memcpy_async");
  auto const is_pinned = dst.is_device_accessible();
  cuda_memcpy_async_impl(dst.data(),
                         src.data(),
                         src.size_bytes(),
                         is_pinned ? host_memory_kind::PINNED : host_memory_kind::PAGEABLE,
                         stream);
}

/**
 * @brief Synchronously copies data from host to device memory.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination device memory
 * @param src Source host memory
 * @param stream CUDA stream used for the copy
 */
template <typename T>
void cuda_memcpy(device_span<T> dst, host_span<T const> src, rmm::cuda_stream_view stream)
{
  cuda_memcpy_async(dst, src, stream);
  stream.synchronize();
}

/**
 * @brief Synchronously copies data from device to host memory.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination host memory
 * @param src Source device memory
 * @param stream CUDA stream used for the copy
 */
template <typename T>
void cuda_memcpy(host_span<T> dst, device_span<T const> src, rmm::cuda_stream_view stream)
{
  cuda_memcpy_async(dst, src, stream);
  stream.synchronize();
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
