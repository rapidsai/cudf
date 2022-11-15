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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/normalize.hpp>

class TextNormalize : public cudf::benchmark {
};

static void BM_normalize(benchmark::State& state, bool to_lower)
{
  auto const n_rows          = static_cast<cudf::size_type>(state.range(0));
  auto const max_str_length  = static_cast<cudf::size_type>(state.range(1));
  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{n_rows}, profile);
  cudf::strings_column_view input(column->view());

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    nvtext::normalize_characters(input, to_lower);
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
      size_t total_chars = static_cast<size_t>(row_count) * rowlen * 4;
      if (total_chars < static_cast<size_t>(std::numeric_limits<cudf::size_type>::max())) {
        b->Args({row_count, rowlen});
      }
    }
  }
}

#define NVTEXT_BENCHMARK_DEFINE(name, lower)             \
  BENCHMARK_DEFINE_F(TextNormalize, name)                \
  (::benchmark::State & st) { BM_normalize(st, lower); } \
  BENCHMARK_REGISTER_F(TextNormalize, name)              \
    ->Apply(generate_bench_args)                         \
    ->UseManualTime()                                    \
    ->Unit(benchmark::kMillisecond);

NVTEXT_BENCHMARK_DEFINE(characters, false)
NVTEXT_BENCHMARK_DEFINE(to_lower, true)
