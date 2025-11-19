/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_find_string(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const hit_rate  = static_cast<cudf::size_type>(state.get_int64("hit_rate"));
  auto const api       = state.get_string("api");
  auto const tgt_type  = state.get_string("target");

  auto const stream = cudf::get_default_stream();
  auto const col    = create_string_column(num_rows, max_width, hit_rate);
  auto const input  = cudf::strings_column_view(col->view());

  auto target        = cudf::string_scalar("0987 5W43");
  auto targets_col   = cudf::make_column_from_scalar(target, num_rows);
  auto const targets = cudf::strings_column_view(targets_col->view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = col->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  if (api == "find") {
    state.add_global_memory_writes<nvbench::int32_t>(input.size());
  } else {
    state.add_global_memory_writes<nvbench::int8_t>(input.size());
  }

  if (api == "find") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::find(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::find(input, targets); });
    }
  } else if (api == "contains") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::contains(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::contains(input, targets); });
    }
  } else if (api == "starts_with") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::starts_with(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::starts_with(input, targets); });
    }
  } else if (api == "ends_with") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::ends_with(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::ends_with(input, targets); });
    }
  }
}

NVBENCH_BENCH(bench_find_string)
  .set_name("find_string")
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("hit_rate", {20, 80})  // percentage
  .add_string_axis("api", {"find", "contains", "starts_with", "ends_with"})
  .add_string_axis("target", {"scalar", "column"});
