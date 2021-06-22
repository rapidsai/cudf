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

#include "string_bench_args.hpp"

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>

static constexpr cudf::size_type default_repeat_times = 16;
static constexpr cudf::size_type min_repeat_times     = -16;
static constexpr cudf::size_type max_repeat_times     = 16;

static void BM_repeat_strings_scalar_times(benchmark::State& state)

{
  auto const n_rows         = static_cast<cudf::size_type>(state.range(0));
  auto const max_str_length = static_cast<cudf::size_type>(state.range(1));

  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table =
    create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows}, table_profile);
  auto const strings_col = cudf::strings_column_view(table->view().column(0));

  for ([[maybe_unused]] auto _ : state) {
    cuda_event_timer raii(state, true, rmm::cuda_stream_default);
    cudf::strings::repeat_strings(strings_col, default_repeat_times);
  }

  state.SetBytesProcessed(state.iterations() * strings_col.chars_size());
}

static void BM_repeat_strings_column_times(benchmark::State& state)

{
  auto const n_rows         = static_cast<cudf::size_type>(state.range(0));
  auto const max_str_length = static_cast<cudf::size_type>(state.range(1));

  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  table_profile.set_distribution_params(
    cudf::type_id::INT32, distribution_id::NORMAL, min_repeat_times, max_repeat_times);
  auto const table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::INT32}, 2, row_count{n_rows}, table_profile);
  auto const strings_col      = cudf::strings_column_view(table->view().column(0));
  auto const repeat_times_col = table->view().column(1);

  for ([[maybe_unused]] auto _ : state) {
    cuda_event_timer raii(state, true, rmm::cuda_stream_default);
    cudf::strings::repeat_strings(strings_col, repeat_times_col);
  }

  state.SetBytesProcessed(state.iterations() *
                          (strings_col.chars_size() + repeat_times_col.size() * sizeof(int32_t)));
}

static void generate_bench_args(benchmark::internal::Benchmark* b)
{
  int const min_rows   = 1 << 8;
  int const max_rows   = 1 << 18;
  int const row_mult   = 4;
  int const min_rowlen = 1 << 4;
  int const max_rowlen = 1 << 8;
  int const len_mult   = 4;
  generate_string_bench_args(b, min_rows, max_rows, row_mult, min_rowlen, max_rowlen, len_mult);
}

class RepeatStrings : public cudf::benchmark {
};

#define REPEAT_STRINGS_SCALAR_TIMES_BENCHMARK_DEFINE(name)          \
  BENCHMARK_DEFINE_F(RepeatStrings, name)                           \
  (::benchmark::State & st) { BM_repeat_strings_scalar_times(st); } \
  BENCHMARK_REGISTER_F(RepeatStrings, name)                         \
    ->Apply(generate_bench_args)                                    \
    ->UseManualTime()                                               \
    ->Unit(benchmark::kMillisecond);

#define REPEAT_STRINGS_COLUMN_TIMES_BENCHMARK_DEFINE(name)          \
  BENCHMARK_DEFINE_F(RepeatStrings, name)                           \
  (::benchmark::State & st) { BM_repeat_strings_column_times(st); } \
  BENCHMARK_REGISTER_F(RepeatStrings, name)                         \
    ->Apply(generate_bench_args)                                    \
    ->UseManualTime()                                               \
    ->Unit(benchmark::kMillisecond);

REPEAT_STRINGS_SCALAR_TIMES_BENCHMARK_DEFINE(repeat_strings_scalar_times)
REPEAT_STRINGS_COLUMN_TIMES_BENCHMARK_DEFINE(repeat_strings_column_times)
