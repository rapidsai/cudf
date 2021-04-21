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
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <algorithm>
#include <random>

struct url_string_generator {
  size_t num_chars;
  std::bernoulli_distribution dist;

  url_string_generator(size_t num_chars, double esc_seq_chance)
    : num_chars{num_chars}, dist{esc_seq_chance}
  {
  }

  std::string operator()(std::mt19937& engine)
  {
    std::string str;
    str.reserve(num_chars);
    while (str.size() < num_chars) {
      if (str.size() < num_chars - 3 && dist(engine)) {
        str += "%20";
      } else {
        str.push_back('a');
      }
    }
    return str;
  }
};

cudf::test::strings_column_wrapper generate_column(cudf::size_type num_rows,
                                                   cudf::size_type chars_per_row,
                                                   double esc_seq_chance)
{
  std::mt19937 engine(1);
  url_string_generator url_gen(chars_per_row, esc_seq_chance);
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  std::generate_n(std::back_inserter(strings), num_rows, [&]() { return url_gen(engine); });
  return cudf::test::strings_column_wrapper(strings.begin(), strings.end());
}

template <int esc_seq_pct>
class UrlDecode : public cudf::benchmark {
};

template <int esc_seq_pct>
void BM_url_decode(benchmark::State& state)
{
  cudf::size_type const num_rows      = state.range(0);
  cudf::size_type const chars_per_row = state.range(1);

  auto column       = generate_column(num_rows, chars_per_row, esc_seq_pct / 100.0);
  auto strings_view = cudf::strings_column_view(column);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    auto result = cudf::strings::url_decode(strings_view);
  }

  state.SetBytesProcessed(state.iterations() * num_rows *
                          (chars_per_row + sizeof(cudf::size_type)));
}

#define URLD_BENCHMARK_DEFINE(name, esc_seq_pct)                      \
  BENCHMARK_TEMPLATE_DEFINE_F(UrlDecode, name, esc_seq_pct)           \
  (::benchmark::State & state) { BM_url_decode<esc_seq_pct>(state); } \
  BENCHMARK_REGISTER_F(UrlDecode, name)                               \
    ->RangeMultiplier(10)                                             \
    ->Ranges({{100, 100000}, {10, 10000}})                            \
    ->Unit(benchmark::kMillisecond)                                   \
    ->UseManualTime();

URLD_BENCHMARK_DEFINE(url_decode_10pct, 10)
URLD_BENCHMARK_DEFINE(url_decode_50pct, 50)
