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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/extract.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

static void bench_extract(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));

  auto groups = static_cast<cudf::size_type>(state.get_int64("groups"));

  std::default_random_engine generator;
  std::uniform_int_distribution<int> words_dist(0, 999);
  std::vector<std::string> samples(100);  // 100 unique rows of data to reuse
  std::generate(samples.begin(), samples.end(), [&]() {
    std::string row;  // build a row of random tokens
    while (static_cast<cudf::size_type>(row.size()) < row_width) {
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
  auto map =
    create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
  auto input = cudf::gather(
    cudf::table_view{{samples_column}}, map->view(), cudf::out_of_bounds_policy::DONT_CHECK);
  cudf::strings_column_view strings_view(input->get_column(0).view());
  auto prog = cudf::strings::regex_program::create(pattern);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  // gather some throughput statistics as well
  auto chars_size = strings_view.chars_size(cudf::get_default_stream());
  state.add_element_count(chars_size, "chars_size");            // number of bytes;
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);   // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);  // all bytes are written

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::strings::extract(strings_view, *prog);
  });
}

NVBENCH_BENCH(bench_extract)
  .set_name("extract")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("groups", {1, 2, 4});
