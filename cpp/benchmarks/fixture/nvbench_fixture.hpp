/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/utilities/error.hpp>

#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <string>

namespace cudf {
namespace detail {
static std::string rmm_mode_param{"--rmm_mode"};  ///< RMM mode command-line parameter name
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
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
  }

  inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

  inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

  inline auto make_arena()
  {
    return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(make_cuda());
  }

  inline auto make_managed_pool()
  {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_managed());
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

  nvbench_base_fixture(int argc, char const* const* argv)
  {
    for (int i = 1; i < argc - 1; ++i) {
      std::string arg = argv[i];
      if (arg == detail::rmm_mode_param) {
        i++;
        rmm_mode = argv[i];
      }
    }

    mr = create_memory_resource(rmm_mode);
    rmm::mr::set_current_device_resource(mr.get());
    std::cout << "RMM memory resource = " << rmm_mode << "\n";
  }

  std::shared_ptr<rmm::mr::device_memory_resource> mr;
  std::string rmm_mode{"pool"};
};

}  // namespace cudf
