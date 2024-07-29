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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/filling.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

std::unique_ptr<cudf::column> build_input_column(cudf::size_type n_rows,
                                                 cudf::size_type row_width,
                                                 int32_t hit_rate)
{
  // build input table using the following data
  auto raw_data = cudf::test::strings_column_wrapper(
                    {
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
                    })
                    .release();

  if (row_width / 32 > 1) {
    std::vector<cudf::column_view> columns;
    for (int i = 0; i < row_width / 32; ++i) {
      columns.push_back(raw_data->view());
    }
    raw_data = cudf::strings::concatenate(cudf::table_view(columns));
  }
  auto data_view = raw_data->view();

  // compute number of rows in n_rows that should match
  auto matches = static_cast<int32_t>(n_rows * hit_rate) / 100;

  // Create a randomized gather-map to build a column out of the strings in data.
  data_profile gather_profile =
    data_profile_builder().cardinality(0).null_probability(0.0).distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, 1, data_view.size() - 1);
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

// longer pattern lengths demand more working memory per string
std::string patterns[] = {"^\\d+ [a-z]+", "[A-Z ]+\\d+ +\\d+[A-Z]+\\d+$", "5W43$"};

static void bench_contains(nvbench::state& state)
{
  auto const n_rows        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width     = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const pattern_index = static_cast<cudf::size_type>(state.get_int64("pattern"));
  auto const hit_rate      = static_cast<cudf::size_type>(state.get_int64("hit_rate"));

  if (static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(row_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
  }

  auto col   = build_input_column(n_rows, row_width, hit_rate);
  auto input = cudf::strings_column_view(col->view());

  auto pattern = patterns[pattern_index];
  auto program = cudf::strings::regex_program::create(pattern);

  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_element_count(chars_size, "chars_size");
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int32_t>(input.size());

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::strings::contains_re(input, *program); });
}

NVBENCH_BENCH(bench_contains)
  .set_name("contains")
  .add_int64_axis("row_width", {32, 64, 128, 256, 512})
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216})
  .add_int64_axis("hit_rate", {50, 100})  // percentage
  .add_int64_axis("pattern", {0, 1, 2});
