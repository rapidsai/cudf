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

#include <nvtext/minhash.hpp>

#include <nvbench/nvbench.cuh>

static void bench_minhash(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width  = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const hash_width = static_cast<cudf::size_type>(state.get_int64("hash_width"));
  auto const parameters = static_cast<cudf::size_type>(state.get_int64("parameters"));
  auto const base64     = state.get_int64("hash_type") == 64;

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  auto const strings_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  cudf::strings_column_view input(strings_table->view().column(0));

  data_profile const param_profile = data_profile_builder().no_validity().distribution(
    cudf::type_to_id<cudf::hash_value_type>(),
    distribution_id::NORMAL,
    0u,
    std::numeric_limits<cudf::hash_value_type>::max());
  auto const param_type = base64 ? cudf::type_id::UINT64 : cudf::type_id::UINT32;
  auto const param_table =
    create_random_table({param_type, param_type}, row_count{parameters}, param_profile);
  auto const parameters_a = param_table->view().column(0);
  auto const parameters_b = param_table->view().column(1);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);  // output are hashes

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = base64 ? nvtext::minhash64(input, 0, parameters_a, parameters_b, hash_width)
                         : nvtext::minhash(input, 0, parameters_a, parameters_b, hash_width);
  });
}

NVBENCH_BENCH(bench_minhash)
  .set_name("minhash")
  .add_int64_axis("num_rows", {15000, 30000, 60000})
  .add_int64_axis("row_width", {6000, 28000, 50000})
  .add_int64_axis("hash_width", {12, 24})
  .add_int64_axis("parameters", {26, 260})
  .add_int64_axis("hash_type", {32, 64});
