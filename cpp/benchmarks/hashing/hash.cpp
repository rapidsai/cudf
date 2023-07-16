/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/hashing.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_hash(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const nulls     = static_cast<bool>(state.get_float64("nulls"));
  auto const hash_name = state.get_string("hash_name");

  data_profile const profile = data_profile_builder().null_probability(nulls);
  auto const data            = create_random_table(
    {cudf::type_id::INT64, cudf::type_id::STRING}, row_count{num_rows}, profile);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if (hash_name == "murmurhash3_x86_32") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::hashing::murmurhash3_x86_32(data->view());
    });
  } else if (hash_name == "md5") {
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::md5(data->view()); });
  } else if (hash_name == "spark_murmurhash3_x86_32") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::hashing::spark_murmurhash3_x86_32(data->view());
    });
  } else {
    state.skip(hash_name + ": unknown hash name");
  }
}

NVBENCH_BENCH(bench_hash)
  .set_name("hashing")
  .add_int64_axis("num_rows", {65536, 16777216})
  .add_float64_axis("nulls", {0.0, 0.1})
  .add_string_axis("hash_name", {"murmurhash3_x86_32", "md5", "spark_murmurhash3_x86_32"});
