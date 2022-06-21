/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace cudf {

/**
 * @brief An RAII class setting up RMM memory pool for `nvbench` benchmarks
 *
 * This is a temporary solution before templated fixtures tests are supported
 * in `nvbench`. Similarly to `cudf::benchmark`, creating this RAII object in
 * each benchmark will ensure that the RAPIDS Memory Manager pool mode is used
 * in benchmarks, which eliminates memory allocation / deallocation performance
 * overhead from the benchmark.
 *
 * Example:
 *
 * void my_benchmark(nvbench::state& state) {
 * cudf::rmm_pool_raii pool_raii;
 * state.exec([](nvbench::launch& launch) {
 *       // benchmark stuff
 *  });
 * }
 *
 * NVBENCH_BENCH(my_benchmark);
 */
class rmm_pool_raii {
 private:
  // memory resource factory helpers
  inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

  inline auto make_pool()
  {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
  }

 public:
  rmm_pool_raii()
  {
    mr = make_pool();
    rmm::mr::set_current_device_resource(mr.get());  // set default resource to pool
  }

  ~rmm_pool_raii()
  {
    rmm::mr::set_current_device_resource(nullptr);
    mr.reset();
  }

 private:
  std::shared_ptr<rmm::mr::device_memory_resource> mr;
};

}  // namespace cudf
