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
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <nvbench/nvbench.cuh>

#include <limits>

static void bench_slice(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const stype     = state.get_string("type");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  cudf::strings_column_view input(column->view());
  auto starts_itr = thrust::constant_iterator<cudf::size_type>(row_width / 4);
  auto starts =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(starts_itr, starts_itr + num_rows);
  auto stops_itr = thrust::constant_iterator<cudf::size_type>(row_width / 3);
  auto stops =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(stops_itr, stops_itr + num_rows);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  // gather some throughput statistics as well
  auto chars_size = input.chars_size(stream);
  state.add_element_count(chars_size, "chars_size");           // number of bytes
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);  // all bytes are read
  auto output_size = (row_width / 3 - row_width / 4) * num_rows;
  state.add_global_memory_writes<nvbench::int8_t>(output_size);

  if (stype == "multi") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::slice_strings(input, starts, stops, stream);
    });
  } else {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::slice_strings(input, row_width / 4, row_width / 3, 1, stream);
    });
  }

  set_throughputs(state);
}

NVBENCH_BENCH(bench_slice)
  .set_name("slice")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("type", {"position", "multi"});
