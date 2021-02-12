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
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <limits>

using algorithm = cudf::strings::detail::replace_algorithm;

class StringReplaceScalar : public cudf::benchmark {
};

template <algorithm alg>
static void BM_replace_scalar(benchmark::State& state, int target_size, int32_t maxrepl)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table =
    create_random_table({cudf::type_id::STRING}, 1, row_count{n_rows}, table_profile);
  cudf::strings_column_view input(table->view().column(0));
  cudf::string_scalar target(std::string(target_size, '+'));
  cudf::string_scalar repl("");

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    cudf::strings::detail::replace<alg>(input, target, repl, maxrepl);
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

#define STRINGS_BENCHMARK_DEFINE(name, alg, tsize, maxrepl)                 \
  BENCHMARK_DEFINE_F(StringReplaceScalar, name)                             \
  (::benchmark::State & st) { BM_replace_scalar<alg>(st, tsize, maxrepl); } \
  BENCHMARK_REGISTER_F(StringReplaceScalar, name)                           \
    ->Apply(generate_bench_args)                                            \
    ->UseManualTime()                                                       \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(replace_scalar_autoalg_single, algorithm::AUTO, 1, -1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_autoalg_single_max1, algorithm::AUTO, 1, 1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_autoalg_multi, algorithm::AUTO, 2, -1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_autoalg_multi_max1, algorithm::AUTO, 2, 1)

// Useful for tuning the automatic algorithm heuristic
STRINGS_BENCHMARK_DEFINE(replace_scalar_charalg_single, algorithm::CHAR_PARALLEL, 1, -1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_charalg_single_max1, algorithm::CHAR_PARALLEL, 1, 1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_charalg_multi, algorithm::CHAR_PARALLEL, 2, -1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_charalg_multi_max1, algorithm::CHAR_PARALLEL, 2, 1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_rowalg_single, algorithm::ROW_PARALLEL, 1, -1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_rowalg_single_max1, algorithm::ROW_PARALLEL, 1, 1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_rowalg_multi, algorithm::ROW_PARALLEL, 2, -1)
STRINGS_BENCHMARK_DEFINE(replace_scalar_rowalg_multi_max1, algorithm::ROW_PARALLEL, 2, 1)
