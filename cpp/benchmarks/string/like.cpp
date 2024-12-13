/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_like(nvbench::state& state)
{
  auto const n_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const hit_rate  = static_cast<int32_t>(state.get_int64("hit_rate"));

  auto col   = create_string_column(n_rows, row_width, hit_rate);
  auto input = cudf::strings_column_view(col->view());

  // This pattern forces reading the entire target string (when matched expected)
  auto pattern = std::string("% 5W4_");  // regex equivalent: ".* 5W4.$"

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  // gather some throughput statistics as well
  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_element_count(chars_size, "chars_size");           // number of bytes;
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);  // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(n_rows);     // writes are BOOL8

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = cudf::strings::like(input, pattern); });
}

NVBENCH_BENCH(bench_like)
  .set_name("strings_like")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("hit_rate", {10, 25, 70, 100});
