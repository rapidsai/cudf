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
#include <cudf_test/column_wrapper.hpp>

#include <limits>

class StringReplace : public cudf::benchmark {
};

enum replace_type { scalar, slice, multi };

static void BM_replace(benchmark::State& state, replace_type rt)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table =
    create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows}, table_profile);
  cudf::strings_column_view input(table->view().column(0));
  cudf::string_scalar target("+");
  cudf::string_scalar repl("");
  cudf::test::strings_column_wrapper targets({"+", "-"});
  cudf::test::strings_column_wrapper repls({"", ""});

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    switch (rt) {
      case scalar: cudf::strings::replace(input, target, repl); break;
      case slice: cudf::strings::replace_slice(input, repl, 1, 10); break;
      case multi:
        cudf::strings::replace(
          input, cudf::strings_column_view(targets), cudf::strings_column_view(repls));
        break;
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
      if (total_chars < std::numeric_limits<cudf::size_type>::max()) {
        b->Args({row_count, rowlen});
      }
    }
  }
}

#define STRINGS_BENCHMARK_DEFINE(name)                              \
  BENCHMARK_DEFINE_F(StringReplace, name)                           \
  (::benchmark::State & st) { BM_replace(st, replace_type::name); } \
  BENCHMARK_REGISTER_F(StringReplace, name)                         \
    ->Apply(generate_bench_args)                                    \
    ->UseManualTime()                                               \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(scalar)
STRINGS_BENCHMARK_DEFINE(slice)
STRINGS_BENCHMARK_DEFINE(multi)
