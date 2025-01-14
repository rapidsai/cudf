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

#include <nvtext/generate_ngrams.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

static void bench_hash_ngrams(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const ngrams    = static_cast<cudf::size_type>(state.get_int64("ngrams"));

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const strings_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  cudf::strings_column_view input(strings_table->view().column(0));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  // output are hashes: approximate total number of hashes
  state.add_global_memory_writes<nvbench::int32_t>(num_rows * ngrams);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::hash_character_ngrams(input, ngrams);
  });
}

NVBENCH_BENCH(bench_hash_ngrams)
  .set_name("hash_ngrams")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {128, 512, 2048})
  .add_int64_axis("num_rows", {16384, 32768, 262144})
  .add_int64_axis("ngrams", {5, 10});
