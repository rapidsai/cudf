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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <limits>
#include <vector>

enum FilterAPI { filter, filter_chars, strip };

class StringFilterChars : public cudf::benchmark {
};

static void BM_filter_chars(benchmark::State& state, FilterAPI api)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{n_rows}, profile);
  cudf::strings_column_view input(column->view());

  auto const types = cudf::strings::string_character_types::SPACE;
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> filter_table{
    {cudf::char_utf8{'a'}, cudf::char_utf8{'c'}}};

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::default_stream_value);
    switch (api) {
      case filter: cudf::strings::filter_characters_of_type(input, types); break;
      case filter_chars: cudf::strings::filter_characters(input, filter_table); break;
      case strip: cudf::strings::strip(input); break;
    }
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
  for (int row_count = min_rows; row_count <= max_rows; row_count *= row_mult) {
    for (int rowlen = min_rowlen; rowlen <= max_rowlen; rowlen *= len_mult) {
      // avoid generating combinations that exceed the cudf column limit
      size_t total_chars = static_cast<size_t>(row_count) * rowlen;
      if (total_chars < static_cast<size_t>(std::numeric_limits<cudf::size_type>::max())) {
        b->Args({row_count, rowlen});
      }
    }
  }
}

#define STRINGS_BENCHMARK_DEFINE(name)                                \
  BENCHMARK_DEFINE_F(StringFilterChars, name)                         \
  (::benchmark::State & st) { BM_filter_chars(st, FilterAPI::name); } \
  BENCHMARK_REGISTER_F(StringFilterChars, name)                       \
    ->Apply(generate_bench_args)                                      \
    ->UseManualTime()                                                 \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(filter)
STRINGS_BENCHMARK_DEFINE(filter_chars)
STRINGS_BENCHMARK_DEFINE(strip)
