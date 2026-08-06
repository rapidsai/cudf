/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @brief Specifies when a host-to-device copy consumes its source memory.
 */
enum class host_source_access_order : uint8_t { STREAM, DURING_API_CALL };

void cuda_memcpy_async_impl(void* dst,
                            void const* src,
                            size_t size,
                            host_memory_kind kind,
                            rmm::cuda_stream_view stream,
                            host_source_access_order source_access_order);

/**
 * @brief Wrapper around cudaMemcpyBatchAsync
 *
 * Uses `cudaMemcpyBatchAsync` on CUDA 13.0+ and `cudaMemcpyAsync` otherwise. By default, reading
 * each source buffer is stream ordered and the source must remain valid until the stream has
 * executed the copy. Passing `host_source_access_order::DURING_API_CALL` guarantees that host
 * sources have been consumed before this function returns, allowing callers to release them
 * immediately.
 *
 * All copies share a single source access order and `cudaMemcpyFlagPreferOverlapWithCompute`.
 * Per-copy attributes are not supported by this wrapper; callers requiring different attributes
 * per copy should call `cudaMemcpyBatchAsync` directly.
 *
 * @param dsts Host pointer to a list of destination pointers.
 * @param srcs Host pointer to a list of source pointers.
 * @param sizes Host pointer to a list of sizes.
 * @param count Size of dsts, srcs, sizes arrays
 * @param stream CUDA stream on which copies are enqueued
 * @param source_access_order When the source buffers may be accessed
 *
 * @note if \p stream is the default stream, this function will fallback to `cudaMemcpyAsync` for
 * each copy.
 */
[[nodiscard]] cudaError_t memcpy_batch_async(
  void* const* dsts,
  void const* const* srcs,
  std::size_t const* sizes,
  std::size_t count,
  rmm::cuda_stream_view stream,
  host_source_access_order source_access_order = host_source_access_order::STREAM);

/**
 * @brief Asynchronously copies a single buffer, wrapping `memcpy_batch_async`.
 *
 * Carries the same source-lifetime requirement as `memcpy_batch_async`.
 *
 * Prefer `cudf::detail::cuda_memcpy_async` for host/device copies involving typed spans.
 * Use this function for device-to-device copies or when a raw `void*` interface is required.
 * The copy direction is inferred from the pointer types (`cudaMemcpyDefault`).
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param count Size in bytes to copy
 * @param stream CUDA stream on which the copy is enqueued
 * @param source_access_order When the source buffer may be accessed
 * @return cudaError_t CUDA error code
 */
[[nodiscard]] cudaError_t memcpy_async(
  void* dst,
  void const* src,
  size_t count,
  rmm::cuda_stream_view stream,
  host_source_access_order source_access_order = host_source_access_order::STREAM);

/**
 * @brief Asynchronously copies data from host to device memory.
 *
 * Implementation may use different strategies depending on the size and type of host data. By
 * default, the source must remain valid until the stream has executed the copy. Pass
 * `host_source_access_order::DURING_API_CALL` when the source may be released immediately after
 * this function returns.
 *
 * @param dst Destination device memory
 * @param src Source host memory
 * @param stream CUDA stream used for the copy
 * @param source_access_order When the source buffer may be accessed
 */
template <typename T>
void cuda_memcpy_async(
  device_span<T> dst,
  host_span<T const> src,
  rmm::cuda_stream_view stream,
  host_source_access_order source_access_order = host_source_access_order::STREAM)
{
  CUDF_EXPECTS(dst.size() == src.size(), "Mismatched sizes in cuda_memcpy_async");
  auto const is_pinned = src.is_device_accessible();
  cuda_memcpy_async_impl(dst.data(),
                         src.data(),
                         src.size_bytes(),
                         is_pinned ? host_memory_kind::PINNED : host_memory_kind::PAGEABLE,
                         stream,
                         source_access_order);
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
                         stream,
                         host_source_access_order::STREAM);
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
