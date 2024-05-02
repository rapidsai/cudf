/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

class StringCombine : public cudf::benchmark {};

static void BM_combine(benchmark::State& state)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{n_rows}, table_profile);
  cudf::strings_column_view input1(table->view().column(0));
  cudf::strings_column_view input2(table->view().column(1));
  cudf::string_scalar separator("+");

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    cudf::strings::concatenate(table->view(), separator);
  }

  state.SetBytesProcessed(state.iterations() * (input1.chars_size(cudf::get_default_stream()) +
                                                input2.chars_size(cudf::get_default_stream())));
}

static void generate_bench_args(benchmark::internal::Benchmark* b)
{
  int const min_rows   = 1 << 12;
  int const max_rows   = 1 << 24;
  int const row_mult   = 8;
  int const min_rowlen = 1 << 4;
  int const max_rowlen = 1 << 11;
  int const len_mult   = 4;
  generate_string_bench_args(b, min_rows, max_rows, row_mult, min_rowlen, max_rowlen, len_mult);
}

#define STRINGS_BENCHMARK_DEFINE(name)          \
  BENCHMARK_DEFINE_F(StringCombine, name)       \
  (::benchmark::State & st) { BM_combine(st); } \
  BENCHMARK_REGISTER_F(StringCombine, name)     \
    ->Apply(generate_bench_args)                \
    ->UseManualTime()                           \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(concat)
