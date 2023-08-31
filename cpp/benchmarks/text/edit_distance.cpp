/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/edit_distance.hpp>

#include <nvbench/nvbench.cuh>

#include <rmm/device_buffer.hpp>

static void bench_edit_distance(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));

  if (static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(row_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
  }

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  auto const strings_table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  cudf::strings_column_view input1(strings_table->view().column(0));
  cudf::strings_column_view input2(strings_table->view().column(1));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input1.chars_size() + input2.chars_size();
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  // output are integers (one per row)
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = nvtext::edit_distance(input1, input2); });
}

NVBENCH_BENCH(bench_edit_distance)
  .set_name("edit_distance")
  .add_int64_axis("num_rows", {1024, 4096, 8192, 16364, 32768, 262144})
  .add_int64_axis("row_width", {8, 16, 32, 64, 128, 256});
