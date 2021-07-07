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

#include "string_bench_args.hpp"

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/strings/extract.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <random>

class StringExtract : public cudf::benchmark {
};

static void BM_extract(benchmark::State& state, int groups)
{
  auto const n_rows   = static_cast<cudf::size_type>(state.range(0));
  auto const n_length = static_cast<cudf::size_type>(state.range(1));

  std::default_random_engine generator;
  std::uniform_int_distribution<int> words_dist(0, 999);

  std::vector<std::string> samples(100);  // 100 unique rows of data to reuse
  std::generate(samples.begin(), samples.end(), [&]() {
    std::string row;  // build a row of random tokens
    while (static_cast<int>(row.size()) < n_length) {
      row += std::to_string(words_dist(generator)) + " ";
    }
    return row;
  });

  std::string pattern;
  while (static_cast<int>(pattern.size()) < groups) {
    pattern += "(\\d+) ";
  }

  std::uniform_int_distribution<int> distribution(0, samples.size() - 1);
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [&](auto idx) { return samples.at(distribution(generator)); });
  cudf::test::strings_column_wrapper input(elements, elements + n_rows);
  cudf::strings_column_view view(input);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto results = cudf::strings::extract(view, pattern);
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

#define STRINGS_BENCHMARK_DEFINE(name, instructions)          \
  BENCHMARK_DEFINE_F(StringExtract, name)                     \
  (::benchmark::State & st) { BM_extract(st, instructions); } \
  BENCHMARK_REGISTER_F(StringExtract, name)                   \
    ->Apply(generate_bench_args)                              \
    ->UseManualTime()                                         \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(small, 2)
STRINGS_BENCHMARK_DEFINE(medium, 10)
STRINGS_BENCHMARK_DEFINE(large, 30)
