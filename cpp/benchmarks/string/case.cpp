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

#include <cudf/strings/case.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

void bench_case(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const encoding  = state.get_string("encoding");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);

  auto col_view = column->view();

  cudf::column::contents ascii_contents;
  if (encoding == "ascii") {
    data_profile ascii_profile = data_profile_builder().no_validity().distribution(
      cudf::type_id::INT8, distribution_id::UNIFORM, 32, 126);  // nice ASCII range
    auto input        = cudf::strings_column_view(col_view);
    auto ascii_column = create_random_column(
      cudf::type_id::INT8,
      row_count{static_cast<cudf::size_type>(input.chars_size(cudf::get_default_stream()))},
      ascii_profile);
    auto ascii_data = ascii_column->view();

    col_view = cudf::column_view(col_view.type(),
                                 col_view.size(),
                                 ascii_data.data<char>(),
                                 col_view.null_mask(),
                                 col_view.null_count(),
                                 0,
                                 {input.offsets()});

    ascii_contents = ascii_column->release();
  }
  auto input = cudf::strings_column_view(col_view);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.add_element_count(input.chars_size(cudf::get_default_stream()), "chars_size");
  state.add_global_memory_reads<nvbench::int8_t>(input.chars_size(cudf::get_default_stream()));
  state.add_global_memory_writes<nvbench::int8_t>(input.chars_size(cudf::get_default_stream()));

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = cudf::strings::to_lower(input); });
}

NVBENCH_BENCH(bench_case)
  .set_name("case")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("encoding", {"ascii", "utf8"});
