/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_find_string(nvbench::state& state)
{
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const index     = static_cast<cudf::size_type>(state.get_int64("index"));

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const input  = cudf::strings_column_view(column->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<nvbench::int8_t>(column->alloc_size());
  state.add_global_memory_writes<nvbench::int32_t>(input.size());

  auto const target = cudf::string_scalar(" ");

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::strings::find_instance(input, target, index); });
}

NVBENCH_BENCH(bench_find_string)
  .set_name("find_instance")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("index", {5, 10});
