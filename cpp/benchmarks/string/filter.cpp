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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

static void bench_filter(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const api       = state.get_string("api");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const input  = cudf::strings_column_view(column->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto chars_size = input.chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);

  if (api == "filter") {
    auto const types = cudf::strings::string_character_types::SPACE;
    {
      auto result = cudf::strings::filter_characters_of_type(input, types);
      auto sv     = cudf::strings_column_view(result->view());
      state.add_global_memory_writes<nvbench::int8_t>(sv.chars_size(stream));
    }
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::filter_characters_of_type(input, types);
    });
  } else if (api == "chars") {
    state.add_global_memory_writes<nvbench::int8_t>(chars_size);
    std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> filter_table{
      {cudf::char_utf8{'a'}, cudf::char_utf8{'c'}}};
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::filter_characters(input, filter_table);
    });
  } else if (api == "strip") {
    {
      auto result = cudf::strings::strip(input);
      auto sv     = cudf::strings_column_view(result->view());
      state.add_global_memory_writes<nvbench::int8_t>(sv.chars_size(stream));
    }
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::strip(input); });
  }
}

NVBENCH_BENCH(bench_filter)
  .set_name("filter")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("api", {"filter", "chars", "strip"});
