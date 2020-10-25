/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>
#include "../../../tests/io/algo/scan_select_if.cuh"

#include <thrust/iterator/constant_iterator.h>

struct simple_op {
  inline constexpr uint32_t operator()(uint32_t lhs, uint32_t rhs) { return lhs + rhs; }
  inline constexpr bool operator()(uint32_t value) { return value % 2 == 0; }
};

static void BM_scan_select_if(benchmark::State& state)
{
  using T             = uint64_t;
  uint32_t input_size = state.range(0);
  auto op             = simple_op{};

  auto input   = thrust::make_constant_iterator<T>(1);
  auto d_input = thrust::device_vector<T>(input, input + input_size);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto d_result = scan_select_if(d_input.begin(), d_input.end(), op, op);
  }

  state.SetBytesProcessed(state.iterations() * input_size * sizeof(T));
}

class ScanSelectIfBenchmark : public cudf::benchmark {
};

#define DUMMY_BM_BENCHMARK_DEFINE(name)                                       \
  BENCHMARK_DEFINE_F(ScanSelectIfBenchmark, name)(::benchmark::State & state) \
  {                                                                           \
    BM_scan_select_if(state);                                                 \
  }                                                                           \
  BENCHMARK_REGISTER_F(ScanSelectIfBenchmark, name)                           \
    ->Ranges({{1 << 10, 1 << 30}})                                            \
    ->UseManualTime()                                                         \
    ->Unit(benchmark::kMillisecond);

DUMMY_BM_BENCHMARK_DEFINE(scan_select_if);
