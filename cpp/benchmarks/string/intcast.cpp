/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/strings/convert/int_cast.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

void bench_intcast(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const from_num = state.get_string("dir") == "from";
  auto const endian   = state.get_string("endian") == "little" ? cudf::strings::endian::LITTLE
                                                               : cudf::strings::endian::BIG;

  data_profile const profile = data_profile_builder()
                                 .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, 8)
                                 .no_validity();
  auto const column    = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const sv        = cudf::strings_column_view(column->view());
  auto const data_type = cudf::data_type{cudf::type_id::INT64};
  auto const numbers   = cudf::strings::cast_to_integer(sv, data_type, endian);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if (from_num) {
    state.add_global_memory_reads<int64_t>(num_rows);
    state.add_global_memory_writes<int8_t>(column->alloc_size());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = cudf::strings::cast_from_integer(numbers->view(), endian);
    });
  } else {
    state.add_global_memory_reads<int8_t>(column->alloc_size());
    state.add_global_memory_writes<int64_t>(num_rows);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = cudf::strings::cast_to_integer(sv, data_type, endian);
    });
  }
}

NVBENCH_BENCH(bench_intcast)
  .set_name("intcast")
  .add_int64_axis("num_rows", {262144, 2097152, 8388608})
  .add_string_axis("endian", {"little", "big"})
  .add_string_axis("dir", {"to", "from"});
