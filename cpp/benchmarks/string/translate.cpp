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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>

class StringTranslate : public cudf::benchmark {};

using entry_type = std::pair<cudf::char_utf8, cudf::char_utf8>;

static void BM_translate(benchmark::State& state, int entry_count)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{n_rows}, profile);
  cudf::strings_column_view input(column->view());

  std::vector<entry_type> entries(entry_count);
  std::transform(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(entry_count),
                 entries.begin(),
                 [](auto idx) -> entry_type {
                   return entry_type{'!' + idx, '~' - idx};
                 });

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    cudf::strings::translate(input, entries);
  }

  state.SetBytesProcessed(state.iterations() * input.chars_size(cudf::get_default_stream()));
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

#define STRINGS_BENCHMARK_DEFINE(name, entries)            \
  BENCHMARK_DEFINE_F(StringTranslate, name)                \
  (::benchmark::State & st) { BM_translate(st, entries); } \
  BENCHMARK_REGISTER_F(StringTranslate, name)              \
    ->Apply(generate_bench_args)                           \
    ->UseManualTime()                                      \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(translate_small, 5)
STRINGS_BENCHMARK_DEFINE(translate_medium, 25)
STRINGS_BENCHMARK_DEFINE(translate_large, 50)
