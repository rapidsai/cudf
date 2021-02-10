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

#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>

class StringContains : public cudf::benchmark {
};

static void BM_contains(benchmark::State& state, bool contains)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto const table = create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows});
  cudf::strings_column_view input(table->view().column(0));

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    // contains_re(), matches_re(), and count_re() all have similar functions
    // with count_re() being the most regex intensive
    contains ? cudf::strings::contains_re(input, "\\d+") : cudf::strings::count_re(input, "\\d+");
  }

  state.SetBytesProcessed(state.iterations() * input.chars_size());
}

#define STRINGS_BENCHMARK_DEFINE(name, b)           \
  BENCHMARK_DEFINE_F(StringContains, name)          \
  (::benchmark::State & st) { BM_contains(st, b); } \
  BENCHMARK_REGISTER_F(StringContains, name)        \
    ->RangeMultiplier(8)                            \
    ->Ranges({{1 << 12, 1 << 24}})                  \
    ->UseManualTime()                               \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(contains_re, true)
STRINGS_BENCHMARK_DEFINE(count_re, false)
