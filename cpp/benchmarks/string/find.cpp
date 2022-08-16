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
#include <cudf/strings/find.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <limits>

enum FindAPI { find, find_multi, contains, starts_with, ends_with };

class StringFindScalar : public cudf::benchmark {
};

static void BM_find_scalar(benchmark::State& state, FindAPI find_api)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table = create_random_table({cudf::type_id::STRING}, row_count{n_rows}, table_profile);
  cudf::strings_column_view input(table->view().column(0));
  cudf::string_scalar target("+");
  cudf::test::strings_column_wrapper targets({"+", "-"});

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::default_stream_value);
    switch (find_api) {
      case find: cudf::strings::find(input, target); break;
      case find_multi:
        cudf::strings::find_multiple(input, cudf::strings_column_view(targets));
        break;
      case contains: cudf::strings::contains(input, target); break;
      case starts_with: cudf::strings::starts_with(input, target); break;
      case ends_with: cudf::strings::ends_with(input, target); break;
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

#define STRINGS_BENCHMARK_DEFINE(name)                    \
  BENCHMARK_DEFINE_F(StringFindScalar, name)              \
  (::benchmark::State & st) { BM_find_scalar(st, name); } \
  BENCHMARK_REGISTER_F(StringFindScalar, name)            \
    ->Apply(generate_bench_args)                          \
    ->UseManualTime()                                     \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(find)
STRINGS_BENCHMARK_DEFINE(find_multi)
STRINGS_BENCHMARK_DEFINE(contains)
STRINGS_BENCHMARK_DEFINE(starts_with)
STRINGS_BENCHMARK_DEFINE(ends_with)
