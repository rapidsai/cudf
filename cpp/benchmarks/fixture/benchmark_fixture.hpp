/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <benchmark/benchmark.h>

namespace cudf {

namespace {
// memory resource factory helpers
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_pool_instance()
{
  static rmm::mr::cuda_memory_resource cuda_mr;
  static auto pool_mr =
    std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
      &cuda_mr, rmm::percent_of_free_device_memory(50));
  return pool_mr;
}
}  // namespace

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
  benchmark() : ::benchmark::Fixture()
  {
    char const* env_iterations = std::getenv("CUDF_BENCHMARK_ITERATIONS");
    if (env_iterations != nullptr) { this->Iterations(std::max(0L, atol(env_iterations))); }
  }

  void SetUp(::benchmark::State const& state) override
  {
    mr = make_pool_instance();
    cudf::set_current_device_resource(mr.get());  // set default resource to pool
  }

  void TearDown(::benchmark::State const& state) override
  {
    // reset default resource to the initial resource
    cudf::set_current_device_resource(nullptr);
    mr.reset();
  }

  // eliminate partial override warnings (see benchmark/benchmark.h)
  void SetUp(::benchmark::State& st) override { SetUp(const_cast<::benchmark::State const&>(st)); }
  void TearDown(::benchmark::State& st) override
  {
    TearDown(const_cast<::benchmark::State const&>(st));
  }

  std::shared_ptr<rmm::mr::device_memory_resource> mr;
};

class memory_stats_logger {
 public:
  memory_stats_logger()
    : existing_mr(cudf::get_current_device_resource_ref()),
      statistics_mr(
        rmm::mr::statistics_resource_adaptor<rmm::device_async_resource_ref>(existing_mr))
  {
    cudf::set_current_device_resource_ref(&statistics_mr);
  }

  ~memory_stats_logger() { cudf::set_current_device_resource_ref(existing_mr); }

  [[nodiscard]] size_t peak_memory_usage() const noexcept
  {
    return statistics_mr.get_bytes_counter().peak;
  }

 private:
  rmm::device_async_resource_ref existing_mr;
  rmm::mr::statistics_resource_adaptor<rmm::device_async_resource_ref> statistics_mr;
};

}  // namespace cudf
