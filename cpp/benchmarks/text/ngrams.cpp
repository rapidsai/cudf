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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/string/string_bench_args.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/generate_ngrams.hpp>

class TextNGrams : public cudf::benchmark {};

enum class ngrams_type { tokens, characters };

static void BM_ngrams(benchmark::State& state, ngrams_type nt)
{
  auto const n_rows          = static_cast<cudf::size_type>(state.range(0));
  auto const max_str_length  = static_cast<cudf::size_type>(state.range(1));
  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{n_rows}, profile);
  cudf::strings_column_view input(column->view());
  auto const separator = cudf::string_scalar("_");

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    switch (nt) {
      case ngrams_type::tokens: nvtext::generate_ngrams(input, 2, separator); break;
      case ngrams_type::characters: nvtext::generate_character_ngrams(input); break;
    }
  }

  state.SetBytesProcessed(state.iterations() * input.chars_size(cudf::get_default_stream()));
}

static void generate_bench_args(benchmark::internal::Benchmark* b)
{
  int const min_rows   = 1 << 12;
  int const max_rows   = 1 << 24;
  int const row_mult   = 8;
  int const min_rowlen = 5;
  int const max_rowlen = 40;
  int const len_mult   = 2;
  generate_string_bench_args(b, min_rows, max_rows, row_mult, min_rowlen, max_rowlen, len_mult);
}

#define NVTEXT_BENCHMARK_DEFINE(name)                             \
  BENCHMARK_DEFINE_F(TextNGrams, name)                            \
  (::benchmark::State & st) { BM_ngrams(st, ngrams_type::name); } \
  BENCHMARK_REGISTER_F(TextNGrams, name)                          \
    ->Apply(generate_bench_args)                                  \
    ->UseManualTime()                                             \
    ->Unit(benchmark::kMillisecond);

NVTEXT_BENCHMARK_DEFINE(tokens)
NVTEXT_BENCHMARK_DEFINE(characters)
