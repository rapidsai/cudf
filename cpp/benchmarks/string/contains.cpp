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

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/contains.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/strings_column_view.hpp>

class StringContains : public cudf::benchmark {
};

enum contains_type { contains, count, findall };

// longer pattern lengths will demand more working memory
std::string patterns[] = {"^\\d+ [a-z]+", "[A-Z ]+\\d+ +\\d+[A-Z]+\\d+$"};

static void BM_contains(benchmark::State& state, contains_type ct)
{
  auto const n_rows        = static_cast<cudf::size_type>(state.range(0));
  auto const pattern_index = static_cast<int32_t>(state.range(1));
  auto const hit_rate      = static_cast<int32_t>(state.range(2));

  std::vector<std::string> rawdata({
    "123 abc 4567890 DEFGHI 0987 5W43",  // matches both patterns;
    "012345 6789 01234 56789 0123 456",  // the rest do not match
    "abc 4567890 DEFGHI 0987 Wxyz 123",
    "abcdefghijklmnopqrstuvwxyz 01234",
    "",
    "AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01",
    "9876543210,abcdefghijklmnopqrstU",
    "9876543210,abcdefghijklmnopqrstU",
    "123 édf 4567890 DéFG 0987 X5",
    "1",
  });
  std::vector<std::string> h_data;
  for (std::size_t i = 0; i < rawdata.size() * rawdata.size(); ++i) {
    h_data.push_back(rawdata[i / rawdata.size()]);
  }
  auto data      = cudf::test::strings_column_wrapper(h_data.begin(), h_data.end());
  auto data_view = cudf::column_view(data);

  auto matches = static_cast<int32_t>(data_view.size() * rawdata.size() / hit_rate);

  // Create a randomized gather-map to build a column out of the strings in data.
  // For hit-rate ~100%, matches is 1 -- only gathers the matching row(s).
  // For hit-rate ~10%, matches is 10 -- use all rows from data (only 10% will match).
  data_profile table_profile;
  table_profile.set_distribution_params(
    cudf::type_id::INT32, distribution_id::UNIFORM, 0, matches - 1);
  table_profile.set_null_frequency(0.0);  // no nulls for gather-map

  auto table      = create_random_table({cudf::type_id::INT32}, row_count{n_rows}, table_profile);
  auto gather_map = cudf::column_view(table->view().column(0));
  table           = cudf::gather(cudf::table_view({data_view}), gather_map);
  auto input      = cudf::strings_column_view(table->view().column(0));

  auto pattern = patterns[pattern_index];

  for (auto _ : state) {
    cuda_event_timer raii(state, true, rmm::cuda_stream_default);
    switch (ct) {
      case contains_type::contains:  // contains_re and matches_re use the same main logic
        cudf::strings::contains_re(input, pattern);
        break;
      case contains_type::count:  // counts occurrences of matches
        cudf::strings::count_re(input, pattern);
        break;
      case contains_type::findall:  // returns occurrences of all matches
        cudf::strings::findall(input, pattern);
        break;
    }
  }

  state.SetBytesProcessed(state.iterations() * input.chars_size());
}

#define STRINGS_BENCHMARK_DEFINE(name, b)                                                   \
  BENCHMARK_DEFINE_F(StringContains, name)                                                  \
  (::benchmark::State & st) { BM_contains(st, contains_type::b); }                          \
  BENCHMARK_REGISTER_F(StringContains, name)                                                \
    ->ArgsProduct({{4096, 32768, 262144, 2097152, 16777216}, /* row count */                \
                   {0, 1},                                   /* patterns index */           \
                   {10, 18, 46, 64, 98}})                    /* 11%, 25%, 50%, 70%, 100% */ \
    ->UseManualTime()                                                                       \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(contains_re, contains)
STRINGS_BENCHMARK_DEFINE(count_re, count)
STRINGS_BENCHMARK_DEFINE(findall_re, findall)
