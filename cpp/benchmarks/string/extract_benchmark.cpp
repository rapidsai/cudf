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

#include <cudf/strings/extract.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/column_wrapper.hpp>

#include "string_bench_args.hpp"

class StringExtract : public cudf::benchmark {
};

static void BM_extract(benchmark::State& state, int re_instructions)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table =
    create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows}, table_profile);
  cudf::strings_column_view input(table->view().column(0));
  std::string const raw_pattern =
    "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
    "5678901234567890123456789012345678901234567890";
  std::string const pattern = "(" + raw_pattern.substr(0, re_instructions) + ")";

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    auto results = cudf::strings::extract(input, pattern);
  }

  state.SetBytesProcessed(state.iterations() * input.chars_size());
}

static void generate_bench_args(benchmark::internal::Benchmark* b)
{
  int const min_rows   = 1 << 12;
  int const max_rows   = 1 << 24;
  int const row_mult   = 8;
  int const min_rowlen = 1 << 5;
  int const max_rowlen = 1 << 13;
  int const len_mult   = 4;
  generate_string_bench_args(b, min_rows, max_rows, row_mult, min_rowlen, max_rowlen, len_mult);
}

#define STRINGS_BENCHMARK_DEFINE(name, instructions)          \
  BENCHMARK_DEFINE_F(StringExtract, name)                     \
  (::benchmark::State & st) { BM_extract(st, instructions); } \
  BENCHMARK_REGISTER_F(StringExtract, name)                   \
    ->Apply(generate_bench_args)                              \
    ->UseManualTime()                                         \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(small, 4)
STRINGS_BENCHMARK_DEFINE(medium, 48)
STRINGS_BENCHMARK_DEFINE(large, 128)
