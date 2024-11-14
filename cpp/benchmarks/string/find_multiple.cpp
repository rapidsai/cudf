/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/strings/find.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_find_string(nvbench::state& state)
{
  auto const n_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width    = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const hit_rate     = static_cast<cudf::size_type>(state.get_int64("hit_rate"));
  auto const target_count = static_cast<cudf::size_type>(state.get_int64("targets"));
  auto const api          = state.get_string("api");

  auto const stream = cudf::get_default_stream();
  auto const col    = create_string_column(n_rows, row_width, hit_rate);
  auto const input  = cudf::strings_column_view(col->view());

  // Note that these all match the first row of the raw_data in create_string_column.
  // This is so the hit_rate can properly accounted for.
  std::vector<std::string> const target_data(
    {" abc", "W43", "0987 5W43", "123 abc", "23 abc", "3 abc", "7 5W43", "87 5W43", "987 5W43"});
  auto h_targets = std::vector<std::string>{};
  for (cudf::size_type i = 0; i < target_count; i++) {
    h_targets.emplace_back(target_data[i % target_data.size()]);
  }
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const chars_size = input.chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  if (api == "find") {
    state.add_global_memory_writes<nvbench::int32_t>(input.size());
  } else {
    state.add_global_memory_writes<nvbench::int8_t>(input.size());
  }

  if (api == "find") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::find_multiple(input, cudf::strings_column_view(targets));
    });
  } else if (api == "contains") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::contains_multiple(input, cudf::strings_column_view(targets));
    });
  }
}

NVBENCH_BENCH(bench_find_string)
  .set_name("find_multiple")
  .add_string_axis("api", {"find", "contains"})
  .add_int64_axis("targets", {10, 20, 40})
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("hit_rate", {20, 80});  // percentage
