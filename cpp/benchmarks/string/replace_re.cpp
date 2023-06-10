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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_replace(nvbench::state& state)
{
  auto const n_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const rtype     = state.get_string("type");

  if (static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(row_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
  }

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{n_rows}, profile);
  cudf::strings_column_view input(column->view());

  auto program = cudf::strings::regex_program::create("(\\d+)");

  auto chars_size = input.chars_size();
  state.add_element_count(chars_size, "chars_size");
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  if (rtype == "backref") {
    auto replacement = std::string("#\\1X");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::replace_with_backrefs(input, *program, replacement);
    });
  } else {
    auto replacement = std::string("77");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::replace_re(input, *program, replacement);
    });
  }
}

NVBENCH_BENCH(bench_replace)
  .set_name("replace_re")
  .add_int64_axis("row_width", {32, 64, 128, 256, 512})
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216})
  .add_string_axis("type", {"replace", "backref"});
