/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/reshape.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/functional>

#include <nvbench/nvbench.cuh>

static void bench_table_to_array(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("columns"));

  data_profile profile = data_profile_builder()
                           .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, 1000)
                           .no_validity();
  std::vector<cudf::type_id> types(num_cols, cudf::type_id::INT32);
  auto input_table = create_random_table(types, row_count{num_rows}, profile);

  auto input_view = input_table->view();
  auto stream     = cudf::get_default_stream();

  rmm::device_buffer output(num_rows * num_cols * sizeof(int32_t), stream);
  auto span = cudf::device_span<cuda::std::byte>(reinterpret_cast<cuda::std::byte*>(output.data()),
                                                 output.size());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int32_t>(num_rows * num_cols);   // all bytes are read
  state.add_global_memory_writes<int32_t>(num_rows * num_cols);  // all bytes are written

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::table_to_array(input_view, span, stream); });
}

NVBENCH_BENCH(bench_table_to_array)
  .set_name("table_to_array")
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216})
  .add_int64_axis("columns", {2, 10, 100});
