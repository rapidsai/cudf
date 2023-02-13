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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/strings/case.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

class StringCase : public cudf::benchmark {
};

static void BM_case(benchmark::State& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  auto const column = create_random_column(cudf::type_id::STRING, row_count{n_rows});
  cudf::strings_column_view input(column->view());

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    cudf::strings::to_lower(input);
  }

  state.SetBytesProcessed(state.iterations() * input.chars_size());
}

#define SORT_BENCHMARK_DEFINE(name)          \
  BENCHMARK_DEFINE_F(StringCase, name)       \
  (::benchmark::State & st) { BM_case(st); } \
  BENCHMARK_REGISTER_F(StringCase, name)     \
    ->RangeMultiplier(8)                     \
    ->Ranges({{1 << 12, 1 << 24}})           \
    ->UseManualTime()                        \
    ->Unit(benchmark::kMillisecond);

SORT_BENCHMARK_DEFINE(to_lower)
