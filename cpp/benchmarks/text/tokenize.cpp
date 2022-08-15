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
#include <benchmarks/string/string_bench_args.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/ngrams_tokenize.hpp>
#include <nvtext/tokenize.hpp>

class TextTokenize : public cudf::benchmark {
};

enum class tokenize_type { single, multi, count, count_multi, ngrams, characters };

static void BM_tokenize(benchmark::State& state, tokenize_type tt)
{
  auto const n_rows                = static_cast<cudf::size_type>(state.range(0));
  auto const max_str_length        = static_cast<cudf::size_type>(state.range(1));
  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const table = create_random_table({cudf::type_id::STRING}, row_count{n_rows}, table_profile);
  cudf::strings_column_view input(table->view().column(0));
  cudf::test::strings_column_wrapper delimiters({" ", "+", "-"});

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::default_stream_value);
    switch (tt) {
      case tokenize_type::single:
        // single whitespace delimiter
        nvtext::tokenize(input);
        break;
      case tokenize_type::multi:
        nvtext::tokenize(input, cudf::strings_column_view(delimiters));
        break;
      case tokenize_type::count:
        // single whitespace delimiter
        nvtext::count_tokens(input);
        break;
      case tokenize_type::count_multi:
        nvtext::count_tokens(input, cudf::strings_column_view(delimiters));
        break;
      case tokenize_type::ngrams:
        // default is bigrams
        nvtext::ngrams_tokenize(input);
        break;
      case tokenize_type::characters:
        // every character becomes a string
        nvtext::character_tokenize(input);
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
  generate_string_bench_args(b, min_rows, max_rows, row_mult, min_rowlen, max_rowlen, len_mult);
}

#define NVTEXT_BENCHMARK_DEFINE(name)                                 \
  BENCHMARK_DEFINE_F(TextTokenize, name)                              \
  (::benchmark::State & st) { BM_tokenize(st, tokenize_type::name); } \
  BENCHMARK_REGISTER_F(TextTokenize, name)                            \
    ->Apply(generate_bench_args)                                      \
    ->UseManualTime()                                                 \
    ->Unit(benchmark::kMillisecond);

NVTEXT_BENCHMARK_DEFINE(single)
NVTEXT_BENCHMARK_DEFINE(multi)
NVTEXT_BENCHMARK_DEFINE(count)
NVTEXT_BENCHMARK_DEFINE(count_multi)
NVTEXT_BENCHMARK_DEFINE(ngrams)
NVTEXT_BENCHMARK_DEFINE(characters)
