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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/normalize.hpp>

#include <nvbench/nvbench.cuh>

static void bench_normalize(nvbench::state& state)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width      = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width      = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const normalize_type = state.get_string("type");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  cudf::strings_column_view input(column->view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  if (normalize_type == "spaces") {
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = nvtext::normalize_spaces(input); });
  } else {
    bool const to_lower = (normalize_type == "to_lower");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = nvtext::normalize_characters(input, to_lower);
    });
  }
}

NVBENCH_BENCH(bench_normalize)
  .set_name("normalize")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("type", {"spaces", "characters", "to_lower"});
