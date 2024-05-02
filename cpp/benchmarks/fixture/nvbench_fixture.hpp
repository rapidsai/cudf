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
#pragma once

#include <cudf/io/memory_resource.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/new_delete_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <string>

namespace cudf {

namespace detail {
static std::string rmm_mode_param{"--rmm_mode"};  ///< RMM mode command-line parameter name
static std::string cuio_host_mem_param{
  "--cuio_host_mem"};  ///< cuio host memory mode parameter name
}  // namespace detail

using rmm::mr::device_memory_resource;
using rmm_pinned_pool_t = rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>;

inline rmm::mr::host_memory_resource* fallback_mr(bool fall_to_pinned)
{
  static rmm::mr::pinned_memory_resource pinned_mr{};
  static rmm::mr::new_delete_resource new_delete_mr{};

  if (fall_to_pinned) {
    return &pinned_mr;
  } else {
    return &new_delete_mr;
  }
}

class pinned_fallback_host_memory_resource {
 private:
  rmm_pinned_pool_t* _pool;
  void* pool_begin_{};
  void* pool_end_{};
  size_t pool_size_{};
  rmm::mr::host_memory_resource* fallback_mr_;

 public:
  pinned_fallback_host_memory_resource(rmm_pinned_pool_t* pool, bool is_fall_back_pinned)
    : _pool(pool), pool_size_{pool->pool_size()}, fallback_mr_(fallback_mr(is_fall_back_pinned))
  {
    // allocate from the pinned pool the full size to figure out
    // our beginning and end address.
    if (pool_size_ != 0) {
      pool_begin_ = pool->allocate(pool_size_);
      pool_end_   = static_cast<void*>(static_cast<uint8_t*>(pool_begin_) + pool_size_);
      pool->deallocate(pool_begin_, pool_size_);
    }
  }

  void* allocate(std::size_t bytes,
                 [[maybe_unused]] std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT)
  {
    if (bytes <= pool_size_) {
      try {
        return _pool->allocate(bytes, alignment);
      } catch (const std::exception& unused) {
      }
    }

    // std::cout << "Falling back!\n";
    return fallback_mr_->allocate(bytes, alignment);
  }
  void deallocate(void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::RMM_DEFAULT_HOST_ALIGNMENT) noexcept
  {
    if (bytes <= pool_size_ && ptr >= pool_begin_ && ptr <= pool_end_) {
      _pool->deallocate(ptr, bytes, alignment);
    } else {
      fallback_mr_->deallocate(ptr, bytes, alignment);
    }
  }

  void* allocate_async(std::size_t bytes, [[maybe_unused]] cuda::stream_ref stream)
  {
    return allocate(bytes);
  }

  void* allocate_async(std::size_t bytes,
                       std::size_t alignment,
                       [[maybe_unused]] cuda::stream_ref stream)
  {
    return allocate(bytes, alignment);
  }

  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        [[maybe_unused]] cuda::stream_ref stream) noexcept
  {
    return deallocate(ptr, bytes);
  }

  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        std::size_t alignment,
                        [[maybe_unused]] cuda::stream_ref stream) noexcept
  {
    return deallocate(ptr, bytes, alignment);
  }
  bool operator==(const pinned_fallback_host_memory_resource&) const { return true; }

  bool operator!=(const pinned_fallback_host_memory_resource&) const { return false; }

  friend void get_property(pinned_fallback_host_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  friend void get_property(pinned_fallback_host_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
  }
};

/**
 * Base fixture for cudf benchmarks using nvbench.
 *
 * Initializes the default memory resource to use the RMM pool device resource.
 */
struct nvbench_base_fixture {
  using host_pooled_mr_t = rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>;

  inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

  inline auto make_pool()
  {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      make_cuda(), rmm::percent_of_free_device_memory(50));
  }

  inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

  inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

  inline auto make_arena()
  {
    return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(make_cuda());
  }

  inline auto make_managed_pool()
  {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      make_managed(), rmm::percent_of_free_device_memory(50));
  }

  inline std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(
    std::string const& mode)
  {
    if (mode == "cuda") return make_cuda();
    if (mode == "pool") return make_pool();
    if (mode == "async") return make_async();
    if (mode == "arena") return make_arena();
    if (mode == "managed") return make_managed();
    if (mode == "managed_pool") return make_managed_pool();
    CUDF_FAIL("Unknown rmm_mode parameter: " + mode +
              "\nExpecting: cuda, pool, async, arena, managed, or managed_pool");
  }

  inline rmm::host_async_resource_ref make_cuio_host_pinned()
  {
    static std::shared_ptr<rmm::mr::pinned_host_memory_resource> mr =
      std::make_shared<rmm::mr::pinned_host_memory_resource>();
    return *mr;
  }

  inline rmm::host_async_resource_ref make_cuio_host_pinned_pool_expandable()
  {
    if (!this->host_pooled_mr) {
      // Don't store in static, as the CUDA context may be destroyed before static destruction
      this->host_pooled_mr = std::make_shared<host_pooled_mr_t>(
        std::make_shared<rmm::mr::pinned_host_memory_resource>().get(),
        size_t{1} * 1024 * 1024 * 1024);
    }

    return *this->host_pooled_mr;
  }

  inline rmm::host_async_resource_ref make_cuio_host_pinned_pool_pinned_fallback(
    std::optional<size_t> size, bool fall_to_pinned)
  {
    using host_pooled_mr = rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>;

    if (not size.has_value()) {
      if (getenv("CUIO_PINNED_POOL_SIZE")) {
        size = atoi(getenv("CUIO_PINNED_POOL_SIZE"));
      } else {
        size_t free{}, total{};
        cudaMemGetInfo(&free, &total);
        size = std::min(total / 200, size_t{100} * 1024 * 1024);
      }
    }

    auto pool_size = (size.value() + 255) & ~255;
    std::cout << "CUIO pinned pool size = " << pool_size << "\n";

    static std::shared_ptr<host_pooled_mr> pool_mr = std::make_shared<host_pooled_mr>(
      std::make_shared<rmm::mr::pinned_host_memory_resource>().get(), pool_size, pool_size);

    static std::shared_ptr<pinned_fallback_host_memory_resource> mr =
      std::make_shared<pinned_fallback_host_memory_resource>(pool_mr.get(), fall_to_pinned);

    return *mr;
  }

  inline rmm::host_async_resource_ref create_cuio_host_memory_resource(std::string const& mode)
  {
    if (mode == "pageable") return make_cuio_host_pinned_pool_pinned_fallback(0, false);
    if (mode == "pinned") return make_cuio_host_pinned_pool_pinned_fallback(0, true);
    if (mode == "pinned_pool_fall_to_pageable")
      return make_cuio_host_pinned_pool_pinned_fallback(std::nullopt, false);
    if (mode == "pinned_pool_fall_to_pinned")
      return make_cuio_host_pinned_pool_pinned_fallback(std::nullopt, true);
    CUDF_FAIL("Unknown cuio_host_mem parameter: " + mode + "\n");
  }

  nvbench_base_fixture(int argc, char const* const* argv)
  {
    for (int i = 1; i < argc - 1; ++i) {
      std::string arg = argv[i];
      if (arg == detail::rmm_mode_param) {
        i++;
        rmm_mode = argv[i];
      } else if (arg == detail::cuio_host_mem_param) {
        i++;
        cuio_host_mode = argv[i];
      }
    }

    mr = create_memory_resource(rmm_mode);
    rmm::mr::set_current_device_resource(mr.get());
    std::cout << "RMM memory resource = " << rmm_mode << "\n";

    cudf::io::set_host_memory_resource(create_cuio_host_memory_resource(cuio_host_mode));
    std::cout << "CUIO host memory resource = " << cuio_host_mode << "\n";
  }

  ~nvbench_base_fixture()
  {
    // Ensure the the pool is freed before the CUDA context is destroyed:
    cudf::io::set_host_memory_resource(this->make_cuio_host_pinned());
  }

  std::shared_ptr<rmm::mr::device_memory_resource> mr;
  std::string rmm_mode{"pool"};

  std::shared_ptr<host_pooled_mr_t> host_pooled_mr;
  std::string cuio_host_mode{"pinned_pool_fall_to_pinned"};
};

}  // namespace cudf
