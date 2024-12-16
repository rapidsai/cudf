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

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_copy_range(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
      .no_validity();
  auto const source_tables = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{num_rows}, table_profile);

  auto const start  = num_rows / 4;
  auto const end    = (num_rows * 3) / 4;
  auto const source = source_tables->view().column(0);
  auto const target = source_tables->view().column(1);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto chars_size = cudf::strings_column_view(target).chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);   // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);  // both columns are similar size

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    [[maybe_unused]] auto result = cudf::copy_range(source, target, start, end, start / 2);
  });
}

NVBENCH_BENCH(bench_copy_range)
  .set_name("copy_range")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152});
