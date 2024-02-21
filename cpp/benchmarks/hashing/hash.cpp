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

#include <cudf/hashing.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <optional>

static void bench_hash(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const nulls    = state.get_float64("nulls");
  // disable null bitmask if probability is exactly 0.0
  bool const no_nulls  = nulls == 0.0;
  auto const hash_name = state.get_string("hash_name");

  data_profile const profile =
    data_profile_builder().null_probability(no_nulls ? std::nullopt : std::optional<double>{nulls});
  auto const data = create_random_table(
    {cudf::type_id::INT64, cudf::type_id::STRING}, row_count{num_rows}, profile);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // collect statistics
  cudf::strings_column_view input(data->get_column(1).view());
  auto const chars_size = input.chars_size(stream);
  // add memory read from string column
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  // add memory read from int64_t column
  state.add_global_memory_reads<nvbench::int64_t>(num_rows);
  // add memory read from bitmaks
  if (!no_nulls) {
    state.add_global_memory_reads<nvbench::int8_t>(2 *
                                                   cudf::bitmask_allocation_size_bytes(num_rows));
  }
  // memory written depends on used hash

  if (hash_name == "murmurhash3_x86_32") {
    state.add_global_memory_writes<nvbench::uint32_t>(num_rows);

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::hashing::murmurhash3_x86_32(data->view());
    });
  } else if (hash_name == "md5") {
    // md5 creates a 32-byte string
    state.add_global_memory_writes<nvbench::int8_t>(32 * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::md5(data->view()); });
  } else if (hash_name == "sha1") {
    // sha1 creates a 40-byte string
    state.add_global_memory_writes<nvbench::int8_t>(40 * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha1(data->view()); });
  } else if (hash_name == "sha224") {
    // sha224 creates a 56-byte string
    state.add_global_memory_writes<nvbench::int8_t>(56 * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha224(data->view()); });
  } else if (hash_name == "sha256") {
    // sha256 creates a 64-byte string
    state.add_global_memory_writes<nvbench::int8_t>(64 * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha256(data->view()); });
  } else if (hash_name == "sha384") {
    // sha384 creates a 96-byte string
    state.add_global_memory_writes<nvbench::int8_t>(96 * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha384(data->view()); });
  } else if (hash_name == "sha512") {
    // sha512 creates a 128-byte string
    state.add_global_memory_writes<nvbench::int8_t>(128 * num_rows);

    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = cudf::hashing::sha512(data->view()); });
  } else {
    state.skip(hash_name + ": unknown hash name");
  }
}

NVBENCH_BENCH(bench_hash)
  .set_name("hashing")
  .add_int64_axis("num_rows", {65536, 16777216})
  .add_float64_axis("nulls", {0.0, 0.1})
  .add_string_axis("hash_name",
                   {"murmurhash3_x86_32", "md5", "sha1", "sha224", "sha256", "sha384", "sha512"});
