/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/getenv_or.hpp"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>

namespace cudf {

namespace {

class fixed_pinned_pool_memory_resource : public rmm::mr::device_memory_resource {
  using upstream_mr    = rmm::mr::pinned_host_memory_resource;
  using host_pooled_mr = rmm::mr::pool_memory_resource<upstream_mr>;

 private:
  upstream_mr upstream_mr_{};
  size_t pool_size_{0};
  // Raw pointer to avoid a segfault when the pool is destroyed on exit
  host_pooled_mr* pool_{nullptr};
  void* pool_begin_{nullptr};
  void* pool_end_{nullptr};
  cuda::stream_ref stream_{cudf::detail::global_cuda_stream_pool().get_stream().value()};

 public:
  fixed_pinned_pool_memory_resource(size_t size)
    :  // rmm requires the pool size to be a multiple of 256 bytes
      pool_size_{rmm::align_up(size, rmm::CUDA_ALLOCATION_ALIGNMENT)},
      pool_{new host_pooled_mr(upstream_mr_, pool_size_, pool_size_)}
  {
    CUDF_LOG_INFO("Pinned pool size = %zu", pool_size_);

    // Allocate full size from the pinned pool to figure out the beginning and end address
    pool_begin_ = pool_->allocate(stream_, pool_size_);
    pool_end_   = static_cast<void*>(static_cast<uint8_t*>(pool_begin_) + pool_size_);
    pool_->deallocate(stream_, pool_begin_, pool_size_);
  }

  // clang-tidy will complain about this function because it is completely
  // unused at runtime and only exist for tag introspection by CCCL, so we
  // ignore linting. This masks a real issue if we ever want to compile with
  // clang, though, which is that the function will actually be compiled out by
  // clang. If cudf were ever to try to support clang as a compile we would
  // need to force the compiler to emit this symbol. The same goes for the
  // other get_property definitions in this file.
  friend void get_property(fixed_pinned_pool_memory_resource const&,  // NOLINT
                           cuda::mr::device_accessible) noexcept
  {
  }

  friend void get_property(fixed_pinned_pool_memory_resource const&,  // NOLINT
                           cuda::mr::host_accessible) noexcept
  {
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    if (bytes <= pool_size_) {
      try {
        return pool_->allocate(stream, bytes);
      } catch (...) {
        // If the pool is exhausted, fall back to the upstream memory resource
      }
    }

    return upstream_mr_.allocate(stream, bytes);
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    if (bytes <= pool_size_ && ptr >= pool_begin_ && ptr < pool_end_) {
      pool_->deallocate(stream, ptr, bytes);
    } else {
      upstream_mr_.deallocate(stream, ptr, bytes);
    }
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    auto const* other_ptr = dynamic_cast<fixed_pinned_pool_memory_resource const*>(&other);
    return other_ptr != nullptr && pool_ == other_ptr->pool_ && stream_ == other_ptr->stream_;
  }
};

static_assert(cuda::mr::resource_with<fixed_pinned_pool_memory_resource,
                                      cuda::mr::device_accessible,
                                      cuda::mr::host_accessible>,
              "Pinned pool mr must be accessible from both host and device");

CUDF_EXPORT rmm::host_device_async_resource_ref& make_default_pinned_mr(
  std::optional<size_t> config_size)
{
  static fixed_pinned_pool_memory_resource mr = [config_size]() {
    auto const size = [&config_size]() -> size_t {
      if (auto const env_val = getenv("LIBCUDF_PINNED_POOL_SIZE"); env_val != nullptr) {
        return std::atol(env_val);
      }

      if (config_size.has_value()) { return *config_size; }

      auto const total = rmm::available_device_memory().second;
      // 0.5% of the total device memory, capped at 100MB
      return std::min(total / 200, size_t{100} * 1024 * 1024);
    }();

    // make the pool with max size equal to the initial size
    return fixed_pinned_pool_memory_resource{size};
  }();

  static rmm::host_device_async_resource_ref mr_ref{mr};
  return mr_ref;
}

CUDF_EXPORT std::mutex& host_mr_mutex()
{
  static std::mutex map_lock;
  return map_lock;
}

// Must be called with the host_mr_mutex mutex held
CUDF_EXPORT rmm::host_device_async_resource_ref& make_host_mr(
  std::optional<pinned_mr_options> const& opts, bool* did_configure = nullptr)
{
  static rmm::host_device_async_resource_ref* mr_ref = nullptr;
  bool configured                                    = false;
  if (mr_ref == nullptr) {
    configured = true;
    mr_ref     = &make_default_pinned_mr(opts ? opts->pool_size : std::nullopt);
  }

  // If the user passed an out param to detect whether this call configured a resource
  // set the result
  if (did_configure != nullptr) { *did_configure = configured; }

  return *mr_ref;
}

// Must be called with the host_mr_mutex mutex held
CUDF_EXPORT rmm::host_device_async_resource_ref& host_mr()
{
  static rmm::host_device_async_resource_ref mr_ref = make_host_mr(std::nullopt);
  return mr_ref;
}

class new_delete_memory_resource {
 public:
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    try {
      return rmm::detail::aligned_host_allocate(
        bytes, alignment, [](std::size_t size) { return ::operator new(size); });
    } catch (std::bad_alloc const& e) {
      CUDF_FAIL("Failed to allocate memory: " + std::string{e.what()}, rmm::out_of_memory);
    }
  }

  void* allocate([[maybe_unused]] cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    rmm::detail::aligned_host_deallocate(
      ptr, bytes, alignment, [](void* ptr) { ::operator delete(ptr); });
  }

  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate_sync(ptr, bytes, alignment);
  }

  bool operator==(new_delete_memory_resource const& other) const { return true; }

  bool operator!=(new_delete_memory_resource const& other) const { return !operator==(other); }

  // NOLINTBEGIN
  friend void get_property(new_delete_memory_resource const&, cuda::mr::host_accessible) noexcept {}
  // NOLINTEND
};

static_assert(cuda::mr::resource_with<new_delete_memory_resource, cuda::mr::host_accessible>,
              "Pageable pool mr must be accessible from the host");

}  // namespace

rmm::host_device_async_resource_ref set_pinned_memory_resource(
  rmm::host_device_async_resource_ref mr)
{
  std::scoped_lock lock{host_mr_mutex()};
  auto last_mr = host_mr();
  host_mr()    = mr;
  return last_mr;
}

rmm::host_device_async_resource_ref get_pinned_memory_resource()
{
  std::scoped_lock lock{host_mr_mutex()};
  return host_mr();
}

bool config_default_pinned_memory_resource(pinned_mr_options const& opts)
{
  std::scoped_lock lock{host_mr_mutex()};
  auto did_configure = false;
  make_host_mr(opts, &did_configure);
  return did_configure;
}

CUDF_EXPORT auto& kernel_pinned_copy_threshold()
{
  // use cudaMemcpyAsync for all pinned copies
  static std::atomic<size_t> threshold = getenv_or("LIBCUDF_KERNEL_PINNED_COPY_THRESHOLD", 0);
  return threshold;
}

void set_kernel_pinned_copy_threshold(size_t threshold)
{
  kernel_pinned_copy_threshold() = threshold;
}

size_t get_kernel_pinned_copy_threshold() { return kernel_pinned_copy_threshold(); }

CUDF_EXPORT auto& allocate_host_as_pinned_threshold()
{
  // use pageable memory for all host allocations
  static std::atomic<size_t> threshold = getenv_or("LIBCUDF_ALLOCATE_HOST_AS_PINNED_THRESHOLD", 0);
  return threshold;
}

void set_allocate_host_as_pinned_threshold(size_t threshold)
{
  allocate_host_as_pinned_threshold() = threshold;
}

size_t get_allocate_host_as_pinned_threshold() { return allocate_host_as_pinned_threshold(); }

namespace detail {

CUDF_EXPORT rmm::host_async_resource_ref get_pageable_memory_resource()
{
  static new_delete_memory_resource mr{};
  static rmm::host_async_resource_ref mr_ref{mr};
  return mr_ref;
}

}  // namespace detail

}  // namespace cudf
