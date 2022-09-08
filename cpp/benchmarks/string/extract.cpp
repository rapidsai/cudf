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

#include "string_bench_args.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/extract.hpp>
#include <cudf/strings/strings_column_view.hpp>

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

  std::string pattern{""};
  while (groups--) {
    pattern += "(\\d+) ";
  }

  cudf::test::strings_column_wrapper samples_column(samples.begin(), samples.end());
  data_profile const profile = data_profile_builder().no_validity().distribution(
    cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0ul, samples.size() - 1);
  auto map = create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{n_rows}, profile);
  auto input = cudf::gather(
    cudf::table_view{{samples_column}}, map->view(), cudf::out_of_bounds_policy::DONT_CHECK);
  cudf::strings_column_view strings_view(input->get_column(0).view());

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto results = cudf::strings::extract(strings_view, pattern);
  }

  state.SetBytesProcessed(state.iterations() * strings_view.chars_size());
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

STRINGS_BENCHMARK_DEFINE(one, 1)
STRINGS_BENCHMARK_DEFINE(two, 2)
STRINGS_BENCHMARK_DEFINE(four, 4)
STRINGS_BENCHMARK_DEFINE(eight, 8)
