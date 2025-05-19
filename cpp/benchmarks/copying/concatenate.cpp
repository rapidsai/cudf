/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

static void bench_concatenate(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls    = static_cast<cudf::size_type>(state.get_float64("nulls"));

  auto input = create_sequence_table(
    cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols), row_count{num_rows}, nulls);
  auto input_columns = input->view();
  auto column_views  = std::vector<cudf::column_view>(input_columns.begin(), input_columns.end());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int64_t>(num_rows * num_cols);
  state.add_global_memory_writes<int64_t>(num_rows * num_cols);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::concatenate(column_views); });
}

NVBENCH_BENCH(bench_concatenate)
  .set_name("concatenate")
  .add_int64_axis("num_rows", {64, 512, 4096, 32768, 262144})
  .add_int64_axis("num_cols", {2, 8, 64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});

static void bench_concatenate_strings(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols  = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const nulls     = static_cast<cudf::size_type>(state.get_float64("nulls"));

  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width)
      .null_probability(nulls);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const input  = column->view();

  auto column_views = std::vector<cudf::column_view>(num_cols, input);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const sv = cudf::strings_column_view(input);
  state.add_global_memory_reads<int8_t>(sv.chars_size(stream) * num_cols);
  state.add_global_memory_writes<int64_t>(sv.chars_size(stream) * num_cols);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::concatenate(column_views); });
}

NVBENCH_BENCH(bench_concatenate_strings)
  .set_name("concatenate_strings")
  .add_int64_axis("num_rows", {256, 512, 4096, 16384})
  .add_int64_axis("num_cols", {2, 8, 64, 256})
  .add_int64_axis("row_width", {32, 128})
  .add_float64_axis("nulls", {0.0, 0.3});
