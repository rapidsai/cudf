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

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

class SortStrings : public cudf::benchmark {
};

static void BM_sort(benchmark::State& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};

  auto const table = create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows});

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    cudf::sort(table->view());
  }
}

#define SORT_BENCHMARK_DEFINE(name)          \
  BENCHMARK_DEFINE_F(SortStrings, name)      \
  (::benchmark::State & st) { BM_sort(st); } \
  BENCHMARK_REGISTER_F(SortStrings, name)    \
    ->RangeMultiplier(8)                     \
    ->Ranges({{1 << 10, 1 << 24}})           \
    ->UseManualTime()                        \
    ->Unit(benchmark::kMillisecond);

SORT_BENCHMARK_DEFINE(stringssort)
