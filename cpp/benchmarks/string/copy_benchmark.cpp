/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <algorithm>
#include <random>

class StringCopy : public cudf::benchmark {
};

static void BM_copy(benchmark::State& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto const source = create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows});
  auto const target = create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows});

  // scatter indices
  std::vector<cudf::size_type> host_map_data(n_rows);
  std::iota(host_map_data.begin(), host_map_data.end(), 0);
  std::random_shuffle(host_map_data.begin(), host_map_data.end());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> scatter_map(host_map_data.begin(),
                                                                      host_map_data.end());

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    cudf::scatter(source->view(), scatter_map, target->view());
  }

  state.SetBytesProcessed(state.iterations() *
                          cudf::strings_column_view(source->view().column(0)).chars_size());
}

#define SORT_BENCHMARK_DEFINE(name)          \
  BENCHMARK_DEFINE_F(StringCopy, name)       \
  (::benchmark::State & st) { BM_copy(st); } \
  BENCHMARK_REGISTER_F(StringCopy, name)     \
    ->RangeMultiplier(8)                     \
    ->Ranges({{1 << 12, 1 << 24}})           \
    ->UseManualTime()                        \
    ->Unit(benchmark::kMillisecond);

SORT_BENCHMARK_DEFINE(scatter)
