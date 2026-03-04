/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// clang-format off
// Forward declaring this type with hidden visibility supersedes the upstream
// declaration and therefore hides instantiations in this file. This prevents
// the specific symbol conflict observed in
// https://github.com/rapidsai/rmm/issues/2219 between nvcomp's instantiation
// of pool_memory_resource<pinned_host_memory_resource> and libcudf's, but it
// does not fix the broader issues around rmm's symbol visibility that are
// raised in that issue. Those will be fixed upstream at a later date.
namespace rmm::mr {
template <typename Upstream>
class pool_memory_resource;
}
// clang-format on

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
#include <cstdlib>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_set>

namespace cudf {

namespace {

class pinned_pool_with_fallback_memory_resource : public rmm::mr::device_memory_resource {
  using upstream_mr    = rmm::mr::pinned_host_memory_resource;
  using host_pooled_mr = rmm::mr::pool_memory_resource<upstream_mr>;

 private:
  upstream_mr upstream_mr_{};
  size_t initial_pool_size_{0};
  size_t max_pool_size_{0};
  // Raw pointer to avoid a segfault when the pool is destroyed on exit
  host_pooled_mr* pool_{nullptr};
  cuda::stream_ref stream_{cudf::detail::global_cuda_stream_pool().get_stream().value()};

  // Hash set to track fallback allocations
  mutable std::shared_mutex fallback_allocations_mutex_;
  std::unordered_set<void*> fallback_allocations_;

 public:
  pinned_pool_with_fallback_memory_resource(size_t initial_size, size_t max_size)
    :  // rmm requires the pool size to be a multiple of 256 bytes
      initial_pool_size_{rmm::align_up(initial_size, rmm::CUDA_ALLOCATION_ALIGNMENT)},
      max_pool_size_{rmm::align_up(max_size, rmm::CUDA_ALLOCATION_ALIGNMENT)},
      pool_{new host_pooled_mr(upstream_mr_, initial_pool_size_, max_pool_size_)}
  {
    CUDF_LOG_INFO(
      "Pinned pool initial size = %zu, max size = %zu", initial_pool_size_, max_pool_size_);
  }

  // clang-tidy will complain about this function because it is completely
  // unused at runtime and only exist for tag introspection by CCCL, so we
  // ignore linting. This masks a real issue if we ever want to compile with
  // clang, though, which is that the function will actually be compiled out by
  // clang. If cudf were ever to try to support clang as a compile we would
  // need to force the compiler to emit this symbol. The same goes for the
  // other get_property definitions in this file.
  friend void get_property(pinned_pool_with_fallback_memory_resource const&,  // NOLINT
                           cuda::mr::device_accessible) noexcept
  {
  }

  friend void get_property(pinned_pool_with_fallback_memory_resource const&,  // NOLINT
                           cuda::mr::host_accessible) noexcept
  {
  }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    try {
      return pool_->allocate(stream, bytes);
    } catch (...) {
      CUDF_LOG_INFO("Pinned pool exhausted, falling back to new pinned allocation for %zu bytes",
                    bytes);
      // fall back to upstream
      auto* ptr = upstream_mr_.allocate(stream, bytes);

      {
        std::unique_lock lock(fallback_allocations_mutex_);
        fallback_allocations_.insert(ptr);
      }

      return ptr;
    }
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    bool is_fallback{false};
    {
      std::shared_lock lock(fallback_allocations_mutex_);
      is_fallback = fallback_allocations_.find(ptr) != fallback_allocations_.end();
    }

    if (is_fallback) {
      {
        std::unique_lock lock(fallback_allocations_mutex_);
        fallback_allocations_.erase(ptr);
      }
      upstream_mr_.deallocate(stream, ptr, bytes);
    } else {
      pool_->deallocate(stream, ptr, bytes);
    }
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    auto const* other_ptr = dynamic_cast<pinned_pool_with_fallback_memory_resource const*>(&other);
    return other_ptr != nullptr && pool_ == other_ptr->pool_ && stream_ == other_ptr->stream_;
  }
};

static_assert(cuda::mr::resource_with<pinned_pool_with_fallback_memory_resource,
                                      cuda::mr::device_accessible,
                                      cuda::mr::host_accessible>,
              "Pinned pool mr must be accessible from both host and device");

CUDF_EXPORT rmm::host_device_async_resource_ref& make_default_pinned_mr(
  std::optional<size_t> config_size)
{
  static pinned_pool_with_fallback_memory_resource mr = [config_size]() {
    auto const initial_size = [&config_size]() -> size_t {
      if (auto const env_val = getenv("LIBCUDF_PINNED_POOL_SIZE"); env_val != nullptr) {
        return std::atol(env_val);
      }

      if (config_size.has_value()) { return *config_size; }

      auto const total = rmm::available_device_memory().second;
      // 0.5% of the total device memory, capped at 64MB
      return std::min(total / 200, size_t{64} * 1024 * 1024);
    }();

    auto const max_size = [&initial_size]() -> size_t {
      if (auto const env_val = getenv("LIBCUDF_PINNED_POOL_MAX_SIZE"); env_val != nullptr) {
        return std::atol(env_val);
      }
      return initial_size * 16;
    }();

    return pinned_pool_with_fallback_memory_resource{initial_size, max_size};
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
