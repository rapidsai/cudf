/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/jaccard.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

static void bench_jaccard(nvbench::state& state)
{
  auto const num_rows        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width       = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width       = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const substring_width = static_cast<cudf::size_type>(state.get_int64("substring_width"));

  data_profile const strings_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
      .no_validity();
  auto const input_table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  cudf::strings_column_view input1(input_table->view().column(0));
  cudf::strings_column_view input2(input_table->view().column(1));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto chars_size = input1.chars_size(stream) + input2.chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::float32_t>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::jaccard_index(input1, input2, substring_width);
  });
}

NVBENCH_BENCH(bench_jaccard)
  .set_name("jaccard")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {128, 512, 1024, 2048})
  .add_int64_axis("num_rows", {32768, 131072, 262144})
  .add_int64_axis("substring_width", {5, 10});
