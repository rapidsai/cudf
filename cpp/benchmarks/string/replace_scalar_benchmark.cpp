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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>

class StringReplaceScalar : public cudf::benchmark {
};

static void BM_replace_scalar(benchmark::State& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.range(0)};
  cudf::size_type const max_str_length{(cudf::size_type)state.range(1)};
  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table =
    create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows}, table_profile);
  cudf::strings_column_view input(table->view().column(0));
  cudf::string_scalar target("+");
  cudf::string_scalar repl(" ");

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    cudf::strings::replace(input, target, repl);
  }

  state.SetBytesProcessed(state.iterations() * input.chars_size());
}

#define STRINGS_BENCHMARK_DEFINE(name)                 \
  BENCHMARK_DEFINE_F(StringReplaceScalar, name)        \
  (::benchmark::State & st) { BM_replace_scalar(st); } \
  BENCHMARK_REGISTER_F(StringReplaceScalar, name)      \
    ->RangeMultiplier(8)                               \
    ->Ranges({{1 << 12, 1 << 18}, {1 << 5, 1 << 13}})  \
    ->UseManualTime()                                  \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(replace_scalar)
