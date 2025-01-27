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

static void bench_copy(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const api       = state.get_string("api");

  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const source =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, table_profile);

  data_profile const map_profile = data_profile_builder().no_validity().distribution(
    cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0, num_rows);
  auto const map_table =
    create_random_table({cudf::type_to_id<cudf::size_type>()}, row_count{num_rows}, map_profile);
  auto const map_view = map_table->view().column(0);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if (api == "gather") {
    auto result =
      cudf::gather(source->view(), map_view, cudf::out_of_bounds_policy::NULLIFY, stream);
    auto chars_size = cudf::strings_column_view(result->view().column(0)).chars_size(stream);
    state.add_global_memory_reads<nvbench::int8_t>(chars_size +
                                                   (map_view.size() * sizeof(cudf::size_type)));
    state.add_global_memory_writes<nvbench::int8_t>(chars_size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::gather(source->view(), map_view, cudf::out_of_bounds_policy::NULLIFY, stream);
    });
  } else if (api == "scatter") {
    auto const target =
      create_random_table({cudf::type_id::STRING}, row_count{num_rows}, table_profile);
    auto result     = cudf::scatter(source->view(), map_view, target->view(), stream);
    auto chars_size = cudf::strings_column_view(result->view().column(0)).chars_size(stream);
    state.add_global_memory_reads<nvbench::int8_t>(chars_size +
                                                   (map_view.size() * sizeof(cudf::size_type)));
    state.add_global_memory_writes<nvbench::int8_t>(chars_size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::scatter(source->view(), map_view, target->view(), stream);
    });
  }
}

NVBENCH_BENCH(bench_copy)
  .set_name("copy")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("api", {"gather", "scatter"});
