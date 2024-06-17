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

#include <cudf/detail/utilities/logger.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {

namespace {
class fixed_pinned_pool_memory_resource {
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
    CUDF_LOG_INFO("Pinned pool size = {}", pool_size_);

    // Allocate full size from the pinned pool to figure out the beginning and end address
    pool_begin_ = pool_->allocate_async(pool_size_, stream_);
    pool_end_   = static_cast<void*>(static_cast<uint8_t*>(pool_begin_) + pool_size_);
    pool_->deallocate_async(pool_begin_, pool_size_, stream_);
  }

  void* allocate_async(std::size_t bytes, std::size_t alignment, cuda::stream_ref stream)
  {
    if (bytes <= pool_size_) {
      try {
        return pool_->allocate_async(bytes, alignment, stream);
      } catch (...) {
        // If the pool is exhausted, fall back to the upstream memory resource
      }
    }

    return upstream_mr_.allocate_async(bytes, alignment, stream);
  }

  void* allocate_async(std::size_t bytes, cuda::stream_ref stream)
  {
    return allocate_async(bytes, rmm::RMM_DEFAULT_HOST_ALIGNMENT, stream);
  }

  void* allocate(std::size_t bytes, std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT)
  {
    auto const result = allocate_async(bytes, alignment, stream_);
    stream_.wait();
    return result;
  }

  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        std::size_t alignment,
                        cuda::stream_ref stream) noexcept
  {
    if (bytes <= pool_size_ && ptr >= pool_begin_ && ptr < pool_end_) {
      pool_->deallocate_async(ptr, bytes, alignment, stream);
    } else {
      upstream_mr_.deallocate_async(ptr, bytes, alignment, stream);
    }
  }

  void deallocate_async(void* ptr, std::size_t bytes, cuda::stream_ref stream) noexcept
  {
    return deallocate_async(ptr, bytes, rmm::RMM_DEFAULT_HOST_ALIGNMENT, stream);
  }

  void deallocate(void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT) noexcept
  {
    deallocate_async(ptr, bytes, alignment, stream_);
    stream_.wait();
  }

  bool operator==(fixed_pinned_pool_memory_resource const& other) const
  {
    return pool_ == other.pool_ and stream_ == other.stream_;
  }

  bool operator!=(fixed_pinned_pool_memory_resource const& other) const
  {
    return !operator==(other);
  }

  friend void get_property(fixed_pinned_pool_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  friend void get_property(fixed_pinned_pool_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
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

}  // namespace cudf
