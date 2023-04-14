/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/string/string_bench_args.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/replace.hpp>

#include <random>

class TextReplace : public cudf::benchmark {};

static void BM_replace(benchmark::State& state)
{
  auto const n_rows   = static_cast<cudf::size_type>(state.range(0));
  auto const n_length = static_cast<cudf::size_type>(state.range(1));

  std::vector<std::string> words{" ",        "one  ",    "two ",       "three ",     "four ",
                                 "five ",    "six  ",    "sevén  ",    "eight ",     "nine ",
                                 "ten   ",   "eleven ",  "twelve ",    "thirteen  ", "fourteen ",
                                 "fifteen ", "sixteen ", "seventeen ", "eighteen ",  "nineteen "};

  std::default_random_engine generator;
  std::uniform_int_distribution<int> tokens_dist(0, words.size() - 1);
  std::string row;  // build a row of random tokens
  while (static_cast<int>(row.size()) < n_length)
    row += words[tokens_dist(generator)];

  std::uniform_int_distribution<int> position_dist(0, 16);

  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [&](auto idx) { return row.c_str() + position_dist(generator); });
  cudf::test::strings_column_wrapper input(elements, elements + n_rows);
  cudf::strings_column_view view(input);

  cudf::test::strings_column_wrapper targets({"one", "two", "sevén", "zero"});
  cudf::test::strings_column_wrapper replacements({"1", "2", "7", "0"});

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    nvtext::replace_tokens(
      view, cudf::strings_column_view(targets), cudf::strings_column_view(replacements));
  }

  state.SetBytesProcessed(state.iterations() * view.chars_size());
}

static void generate_bench_args(benchmark::internal::Benchmark* b)
{
  int const min_rows          = 1 << 12;
  int const max_rows          = 1 << 24;
  int const row_multiplier    = 8;
  int const min_row_length    = 1 << 5;
  int const max_row_length    = 1 << 13;
  int const length_multiplier = 4;
  generate_string_bench_args(
    b, min_rows, max_rows, row_multiplier, min_row_length, max_row_length, length_multiplier);
}

#define NVTEXT_BENCHMARK_DEFINE(name)           \
  BENCHMARK_DEFINE_F(TextReplace, name)         \
  (::benchmark::State & st) { BM_replace(st); } \
  BENCHMARK_REGISTER_F(TextReplace, name)       \
    ->Apply(generate_bench_args)                \
    ->UseManualTime()                           \
    ->Unit(benchmark::kMillisecond);

NVTEXT_BENCHMARK_DEFINE(replace)
