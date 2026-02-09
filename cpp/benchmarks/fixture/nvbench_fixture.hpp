/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/context.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <string>

namespace cudf {

namespace detail {
static std::string rmm_mode_param{"--rmm_mode"};  ///< RMM mode command-line parameter name
static std::string cuio_host_mem_param{
  "--cuio_host_mem"};  ///< cuio host memory mode parameter name
}  // namespace detail

/**
 * Base fixture for cudf benchmarks using nvbench.
 *
 * Initializes the default memory resource to use the RMM pool device resource.
 */
struct nvbench_base_fixture {
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

  inline rmm::host_device_async_resource_ref make_cuio_host_pinned()
  {
    static std::shared_ptr<rmm::mr::pinned_host_memory_resource> mr =
      std::make_shared<rmm::mr::pinned_host_memory_resource>();
    return *mr;
  }

  inline rmm::host_device_async_resource_ref create_cuio_host_memory_resource(
    std::string const& mode)
  {
    if (mode == "pinned") return make_cuio_host_pinned();
    if (mode == "pinned_pool") return cudf::get_pinned_memory_resource();
    CUDF_FAIL("Unknown cuio_host_mem parameter: " + mode + "\nExpecting: pinned or pinned_pool");
  }

  nvbench_base_fixture(int argc, char const* const* argv)
  {
    cudf::initialize(cudf::init_flags::ALL);

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
    cudf::set_current_device_resource(mr.get());
    std::cout << "RMM memory resource = " << rmm_mode << "\n";

    cudf::set_pinned_memory_resource(create_cuio_host_memory_resource(cuio_host_mode));
    std::cout << "CUIO host memory resource = " << cuio_host_mode << "\n";
  }

  ~nvbench_base_fixture()
  {
    // Ensure the the pool is freed before the CUDA context is destroyed:
    cudf::set_pinned_memory_resource(this->make_cuio_host_pinned());
  }

  std::shared_ptr<rmm::mr::device_memory_resource> mr;
  std::string rmm_mode{"pool"};

  std::string cuio_host_mode{"pinned_pool"};
};

}  // namespace cudf
