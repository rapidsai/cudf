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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/filling.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

class StringContains : public cudf::benchmark {
};

std::unique_ptr<cudf::column> build_input_column(cudf::size_type n_rows, int32_t hit_rate)
{
  // build input table using the following data
  auto data      = cudf::test::strings_column_wrapper({
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
  auto data_view = cudf::column_view(data);

  // compute number of rows in n_rows that should match
  auto matches = static_cast<int32_t>(n_rows * hit_rate) / 100;

  // Create a randomized gather-map to build a column out of the strings in data.
  data_profile gather_profile;
  gather_profile.set_distribution_params(
    cudf::type_id::INT32, distribution_id::UNIFORM, 1, data_view.size() - 1);
  gather_profile.set_null_frequency(0.0);  // no nulls for gather-map
  gather_profile.set_cardinality(0);
  auto gather_table =
    create_random_table({cudf::type_id::INT32}, row_count{n_rows}, gather_profile);
  gather_table->get_column(0).set_null_mask(rmm::device_buffer{}, 0);

  // Create scatter map by placing 0-index values throughout the gather-map
  auto scatter_data = cudf::sequence(
    matches, cudf::numeric_scalar<int32_t>(0), cudf::numeric_scalar<int32_t>(n_rows / matches));
  auto zero_scalar = cudf::numeric_scalar<int32_t>(0);
  auto table       = cudf::scatter({zero_scalar}, scatter_data->view(), gather_table->view());
  auto gather_map  = table->view().column(0);
  table            = cudf::gather(cudf::table_view({data_view}), gather_map);

  return std::move(table->release().front());
}

enum contains_type { contains, count, findall };

// longer pattern lengths demand more working memory per string
std::string patterns[] = {"^\\d+ [a-z]+", "[A-Z ]+\\d+ +\\d+[A-Z]+\\d+$"};

static void BM_contains(benchmark::State& state, contains_type ct)
{
  auto const n_rows        = static_cast<cudf::size_type>(state.range(0));
  auto const pattern_index = static_cast<int32_t>(state.range(1));
  auto const hit_rate      = static_cast<int32_t>(state.range(2));

  auto col   = build_input_column(n_rows, hit_rate);
  auto input = cudf::strings_column_view(col->view());

  auto pattern = patterns[pattern_index];

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::default_stream_value);
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

#define STRINGS_BENCHMARK_DEFINE(name, b)                                         \
  BENCHMARK_DEFINE_F(StringContains, name)                                        \
  (::benchmark::State & st) { BM_contains(st, contains_type::b); }                \
  BENCHMARK_REGISTER_F(StringContains, name)                                      \
    ->ArgsProduct({{4096, 32768, 262144, 2097152, 16777216}, /* row count */      \
                   {0, 1},                                   /* patterns index */ \
                   {1, 5, 10, 25, 70, 100}})                 /* hit rate */       \
    ->UseManualTime()                                                             \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(contains_re, contains)
STRINGS_BENCHMARK_DEFINE(count_re, count)
STRINGS_BENCHMARK_DEFINE(findall_re, findall)
