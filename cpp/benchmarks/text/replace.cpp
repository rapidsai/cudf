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

#include <benchmarks/fixture/benchmark_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/replace.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

static void bench_replace(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));

  std::vector<std::string> words{" ",        "one  ",    "two ",       "three ",     "four ",
                                 "five ",    "six  ",    "sevén  ",    "eight ",     "nine ",
                                 "ten   ",   "eleven ",  "twelve ",    "thirteen  ", "fourteen ",
                                 "fifteen ", "sixteen ", "seventeen ", "eighteen ",  "nineteen "};

  std::default_random_engine generator;
  std::uniform_int_distribution<int> tokens_dist(0, words.size() - 1);
  std::string row;  // build a row of random tokens
  while (static_cast<cudf::size_type>(row.size()) < row_width)
    row += words[tokens_dist(generator)];

  std::uniform_int_distribution<int> position_dist(0, 16);

  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [&](auto idx) { return row.c_str() + position_dist(generator); });
  cudf::test::strings_column_wrapper input(elements, elements + num_rows);
  cudf::strings_column_view view(input);

  cudf::test::strings_column_wrapper targets({"one", "two", "sevén", "zero"});
  cudf::test::strings_column_wrapper replacements({"1", "2", "7", "0"});

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = view.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::replace_tokens(
      view, cudf::strings_column_view(targets), cudf::strings_column_view(replacements));
  });
}

NVBENCH_BENCH(bench_replace)
  .set_name("replace")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152});
