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

#include <cudf/reshape.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_table_to_array(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("columns"));

  if (static_cast<std::size_t>(num_rows) * num_cols >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Input size exceeds cudf::size_type limit");
  }

  data_profile profile =
    data_profile_builder().distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, 1000);
  profile.set_null_probability(0.0);
  std::vector<cudf::type_id> types(num_cols, cudf::type_id::INT32);
  auto input_table = create_random_table(types, row_count{num_rows}, profile);

  auto input_view = input_table->view();
  auto stream     = cudf::get_default_stream();
  auto dtype      = cudf::data_type{cudf::type_id::INT32};

  rmm::device_buffer output(num_rows * num_cols * sizeof(int32_t), stream);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int32_t>(num_rows * num_cols);   // all bytes are read
  state.add_global_memory_writes<int32_t>(num_rows * num_cols);  // all bytes are written

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::table_to_array(input_view, output.data(), dtype, stream);
  });
}

NVBENCH_BENCH(bench_table_to_array)
  .set_name("table_to_array")
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216})
  .add_int64_axis("columns", {2, 10, 100});
