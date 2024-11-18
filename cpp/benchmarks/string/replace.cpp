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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

enum replace_type { scalar, slice, multi };

static void bench_replace(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const api       = state.get_string("api");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);

  cudf::strings_column_view input(column->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const chars_size = input.chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  if (api == "scalar") {
    cudf::string_scalar target("+");
    cudf::string_scalar repl("-");
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::replace(input, target, repl); });
  } else if (api == "multi") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::test::strings_column_wrapper targets({"+", " "});
      cudf::test::strings_column_wrapper repls({"-", "_"});
      cudf::strings::replace_multiple(
        input, cudf::strings_column_view(targets), cudf::strings_column_view(repls));
    });
  } else if (api == "slice") {
    cudf::string_scalar repl("0123456789");
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::replace_slice(input, repl, 1, 10); });
  }
}

NVBENCH_BENCH(bench_replace)
  .set_name("replace")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("api", {"scalar", "multi", "slice"});
