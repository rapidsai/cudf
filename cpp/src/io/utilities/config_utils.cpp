/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "config_utils.hpp"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/io/memory_resource.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdlib>
#include <string>

namespace cudf::io {

namespace detail {

namespace cufile_integration {

namespace {
/**
 * @brief Defines which cuFile usage to enable.
 */
enum class usage_policy : uint8_t { OFF, GDS, ALWAYS, KVIKIO };

/**
 * @brief Get the current usage policy.
 */
usage_policy get_env_policy()
{
  static auto const env_val = getenv_or<std::string>("LIBCUDF_CUFILE_POLICY", "KVIKIO");
  if (env_val == "OFF") return usage_policy::OFF;
  if (env_val == "GDS") return usage_policy::GDS;
  if (env_val == "ALWAYS") return usage_policy::ALWAYS;
  if (env_val == "KVIKIO") return usage_policy::KVIKIO;
  CUDF_FAIL("Invalid LIBCUDF_CUFILE_POLICY value: " + env_val);
}
}  // namespace

bool is_always_enabled() { return get_env_policy() == usage_policy::ALWAYS; }

bool is_gds_enabled() { return is_always_enabled() or get_env_policy() == usage_policy::GDS; }

bool is_kvikio_enabled() { return get_env_policy() == usage_policy::KVIKIO; }

}  // namespace cufile_integration

namespace nvcomp_integration {

namespace {
/**
 * @brief Defines which nvCOMP usage to enable.
 */
enum class usage_policy : uint8_t { OFF, STABLE, ALWAYS };

/**
 * @brief Get the current usage policy.
 */
usage_policy get_env_policy()
{
  static auto const env_val = getenv_or<std::string>("LIBCUDF_NVCOMP_POLICY", "STABLE");
  if (env_val == "OFF") return usage_policy::OFF;
  if (env_val == "STABLE") return usage_policy::STABLE;
  if (env_val == "ALWAYS") return usage_policy::ALWAYS;
  CUDF_FAIL("Invalid LIBCUDF_NVCOMP_POLICY value: " + env_val);
}
}  // namespace

bool is_all_enabled() { return get_env_policy() == usage_policy::ALWAYS; }

bool is_stable_enabled() { return is_all_enabled() or get_env_policy() == usage_policy::STABLE; }

}  // namespace nvcomp_integration

}  // namespace detail

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
    : pool_size_{size}, pool_{new host_pooled_mr(upstream_mr_, size, size)}
  {
    if (pool_size_ == 0) { return; }

    // Allocate full size from the pinned pool to figure out the beginning and end address
    pool_begin_ = pool_->allocate_async(pool_size_, stream_);
    pool_end_   = static_cast<void*>(static_cast<uint8_t*>(pool_begin_) + pool_size_);
    pool_->deallocate_async(pool_begin_, pool_size_, stream_);
  }

  void* do_allocate_async(std::size_t bytes, std::size_t alignment, cuda::stream_ref stream)
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

  void do_deallocate_async(void* ptr,
                           std::size_t bytes,
                           std::size_t alignment,
                           cuda::stream_ref stream) noexcept
  {
    if (bytes <= pool_size_ && ptr >= pool_begin_ && ptr <= pool_end_) {
      pool_->deallocate_async(ptr, bytes, alignment, stream);
    } else {
      upstream_mr_.deallocate_async(ptr, bytes, alignment, stream);
    }
  }

  void* allocate_async(std::size_t bytes, cuda::stream_ref stream)
  {
    return do_allocate_async(bytes, rmm::RMM_DEFAULT_HOST_ALIGNMENT, stream);
  }

  void* allocate_async(std::size_t bytes, std::size_t alignment, cuda::stream_ref stream)
  {
    return do_allocate_async(bytes, alignment, stream);
  }

  void* allocate(std::size_t bytes, std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT)
  {
    auto const result = do_allocate_async(bytes, alignment, stream_);
    stream_.wait();
    return result;
  }

  void deallocate_async(void* ptr, std::size_t bytes, cuda::stream_ref stream) noexcept
  {
    return do_deallocate_async(ptr, bytes, rmm::RMM_DEFAULT_HOST_ALIGNMENT, stream);
  }

  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        std::size_t alignment,
                        cuda::stream_ref stream) noexcept
  {
    return do_deallocate_async(ptr, bytes, alignment, stream);
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

  [[maybe_unused]] friend void get_property(fixed_pinned_pool_memory_resource const&,
                                            cuda::mr::device_accessible) noexcept
  {
  }

  [[maybe_unused]] friend void get_property(fixed_pinned_pool_memory_resource const&,
                                            cuda::mr::host_accessible) noexcept
  {
  }
};

static_assert(cuda::mr::resource_with<fixed_pinned_pool_memory_resource,
                                      cuda::mr::device_accessible,
                                      cuda::mr::host_accessible>,
              "");

}  // namespace

CUDF_EXPORT rmm::host_async_resource_ref& make_default_pinned_mr(std::optional<size_t> config_size)
{
  static fixed_pinned_pool_memory_resource mr = [config_size]() {
    auto const size = [&config_size]() -> size_t {
      if (auto const env_val = getenv("LIBCUDF_PINNED_POOL_SIZE"); env_val != nullptr) {
        return std::atol(env_val);
      }

      if (config_size.has_value()) { return *config_size; }

      size_t free{}, total{};
      CUDF_CUDA_TRY(cudaMemGetInfo(&free, &total));
      // 0.5% of the total device memory, capped at 100MB
      return std::min(total / 200, size_t{100} * 1024 * 1024);
    }();

    // rmm requires the pool size to be a multiple of 256 bytes
    auto const aligned_size = (size + 255) & ~255;
    CUDF_LOG_INFO("Pinned pool size = {}", aligned_size);

    // make the pool with max size equal to the initial size
    return fixed_pinned_pool_memory_resource{aligned_size};
  }();

  static rmm::host_async_resource_ref mr_ref{mr};
  return mr_ref;
}

CUDF_EXPORT std::mutex& host_mr_mutex()
{
  static std::mutex map_lock;
  return map_lock;
}

// Must be called with the host_mr_mutex mutex held
CUDF_EXPORT rmm::host_async_resource_ref& make_host_mr(std::optional<host_mr_options> const& opts)
{
  static rmm::host_async_resource_ref* mr_ref = nullptr;
  if (mr_ref == nullptr) {
    mr_ref = &make_default_pinned_mr(opts ? opts->pool_size : std::nullopt);
  } else {
    // Throw an error if the user tries to reconfigure the default host resource
    CUDF_EXPECTS(opts == std::nullopt, "The default host memory resource has already been created");
  }

  return *mr_ref;
}

// Must be called with the host_mr_mutex mutex held
CUDF_EXPORT rmm::host_async_resource_ref& host_mr()
{
  static rmm::host_async_resource_ref mr_ref = make_host_mr(std::nullopt);
  return mr_ref;
}

rmm::host_async_resource_ref set_host_memory_resource(rmm::host_async_resource_ref mr)
{
  std::scoped_lock lock{host_mr_mutex()};
  auto last_mr = host_mr();
  host_mr()    = mr;
  return last_mr;
}

rmm::host_async_resource_ref get_host_memory_resource()
{
  std::scoped_lock lock{host_mr_mutex()};
  return host_mr();
}

void config_default_host_memory_resource(host_mr_options const& opts)
{
  std::scoped_lock lock{host_mr_mutex()};
  make_host_mr(opts);
}

}  // namespace cudf::io
