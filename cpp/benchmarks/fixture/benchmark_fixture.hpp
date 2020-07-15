/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <benchmark/benchmark.h>
#include "rmm/mr/device/cnmem_memory_resource.hpp"
#include "rmm/mr/device/default_memory_resource.hpp"

namespace cudf {
/**
 * @brief Google Benchmark fixture for libcudf benchmarks
 *
 * libcudf benchmarks should use a fixture derived from this fixture class to
 * ensure that the RAPIDS Memory Manager pool mode is used in benchmarks, which
 * eliminates memory allocation / deallocation performance overhead from the
 * benchmark.
 *
 * The SetUp and TearDown methods of this fixture initialize RMM into pool mode
 * and finalize it, respectively. These methods are called automatically by
 * Google Benchmark
 *
 * Example:
 *
 * template <class T>
 * class my_benchmark : public cudf::benchmark {
 * public:
 *   using TypeParam = T;
 * };
 *
 * Then:
 *
 * BENCHMARK_TEMPLATE_DEFINE_F(my_benchmark, my_test_name, int)
 *   (::benchmark::State& state) {
 *     for (auto _ : state) {
 *       // benchmark stuff
 *     }
 * }
 *
 * BENCHMARK_REGISTER_F(my_benchmark, my_test_name)->Range(128, 512);
 */
class benchmark : public ::benchmark::Fixture {
 public:
  virtual void SetUp(const ::benchmark::State& state)
  {
    auto mr = new rmm::mr::cnmem_memory_resource;
    rmm::mr::set_default_resource(mr);  // set default resource to cnmem
  }

  virtual void TearDown(const ::benchmark::State& state)
  {
    delete rmm::mr::get_default_resource();
    rmm::mr::set_default_resource(nullptr);  // reset default resource to the initial resource
  }

  // eliminate partial override warnings (see benchmark/benchmark.h)
  virtual void SetUp(::benchmark::State& st) { SetUp(const_cast<const ::benchmark::State&>(st)); }
  virtual void TearDown(::benchmark::State& st)
  {
    TearDown(const_cast<const ::benchmark::State&>(st));
  }
};

};  // namespace cudf
