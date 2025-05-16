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
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/generate_ngrams.hpp>

#include <nvbench/nvbench.cuh>

static void bench_ngrams(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width  = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const ngram_type = state.get_string("type");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  cudf::strings_column_view input(column->view());
  auto const separator = cudf::string_scalar("_");

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size * 2);

  if (ngram_type == "chars") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = nvtext::generate_character_ngrams(input);
    });
  } else {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = nvtext::generate_ngrams(input, 2, separator);
    });
  }
}

NVBENCH_BENCH(bench_ngrams)
  .set_name("ngrams")
  .add_int64_axis("num_rows", {131072, 262144, 524288, 1048578})
  .add_int64_axis("row_width", {10, 20, 40, 100})
  .add_string_axis("type", {"chars", "tokens"});
