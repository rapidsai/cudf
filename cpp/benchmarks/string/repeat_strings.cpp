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

#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_repeat(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width  = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width  = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const min_repeat = static_cast<cudf::size_type>(state.get_int64("min_repeat"));
  auto const max_repeat = static_cast<cudf::size_type>(state.get_int64("max_repeat"));
  auto const api        = state.get_string("api");

  auto builder = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  builder.distribution(cudf::type_id::INT32, distribution_id::NORMAL, min_repeat, max_repeat);

  auto const table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::INT32}, row_count{num_rows}, data_profile{builder});
  auto const input = cudf::strings_column_view(table->view().column(0));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto chars_size = input.chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);

  if (api == "scalar") {
    state.add_global_memory_writes<nvbench::int8_t>(chars_size * max_repeat);
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::repeat_strings(input, max_repeat); });
  } else if (api == "column") {
    auto repeats = table->view().column(1);
    {
      auto result = cudf::strings::repeat_strings(input, repeats);
      auto output = cudf::strings_column_view(result->view());
      state.add_global_memory_writes<nvbench::int8_t>(output.chars_size(stream));
    }
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::repeat_strings(input, repeats); });
  }
}

NVBENCH_BENCH(bench_repeat)
  .set_name("repeat")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("min_repeat", {0})
  .add_int64_axis("max_repeat", {16})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("api", {"scalar", "column"});
