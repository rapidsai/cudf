/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudf/aggregation.hpp"
#include "cudf/reduction.hpp"
#include "cudf/table/table.hpp"
#include "cudf/table/table_view.hpp"
#include "cudf/types.hpp"

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <memory>
#include <vector>

static std::unique_ptr<cudf::table> make_bench_table(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls    = static_cast<cudf::size_type>(state.get_float64("nulls"));

  auto table = create_sequence_table(
    cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols), row_count{num_rows}, nulls);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_global_memory_reads<int64_t>(num_rows * num_cols);
  state.add_global_memory_writes<int64_t>(num_rows * num_cols);

  return table;
}

void column_sum(const cudf::column_view& col_view)
{
  auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto result  = cudf::reduce(col_view, *sum_agg, cudf::data_type{cudf::type_id::INT64});
}

// Device Pack and Unpack
static void bench_device_pack(nvbench::state& state)
{
  const auto table      = make_bench_table(state);
  const auto table_view = table->view();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto packed = cudf::pack(table_view); });
}

NVBENCH_BENCH(bench_device_pack)
  .set_name("device_pack")
  .add_int64_axis("num_rows", {4096, 32768, 262144})
  .add_int64_axis("num_cols", {64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});

static void bench_device_unpack(nvbench::state& state)
{
  const auto table      = make_bench_table(state);
  const auto table_view = table->view();
  auto packed           = cudf::pack(table_view);
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto unpacked = cudf::unpack(packed); });
}

NVBENCH_BENCH(bench_device_unpack)
  .set_name("device_unpack")
  .add_int64_axis("num_rows", {4096, 32768, 262144})
  .add_int64_axis("num_cols", {64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});

static void bench_device_unpack_and_column_access(nvbench::state& state)
{
  const auto table      = make_bench_table(state);
  const auto table_view = table->view();
  auto packed           = cudf::pack(table_view);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto unpacked = cudf::unpack(packed);
    column_sum(unpacked.column(0));
  });
}

NVBENCH_BENCH(bench_device_unpack_and_column_access)
  .set_name("device_unpack_and_column_access")
  .add_int64_axis("num_rows", {4096, 32768, 262144})
  .add_int64_axis("num_cols", {64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});

// Host Pack and Unpack
static void bench_host_pack(nvbench::state& state)
{
  const auto table      = make_bench_table(state);
  const auto table_view = table->view();
  auto stream           = cudf::get_default_stream();
  rmm::mr::pinned_host_memory_resource phmr;
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto packed = cudf::pack(table_view, stream, phmr); });
}

NVBENCH_BENCH(bench_host_pack)
  .set_name("host_pack")
  .add_int64_axis("num_rows", {4096, 32768, 262144})
  .add_int64_axis("num_cols", {64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});

static void bench_host_unpack(nvbench::state& state)
{
  const auto table      = make_bench_table(state);
  const auto table_view = table->view();
  auto stream           = cudf::get_default_stream();
  rmm::mr::pinned_host_memory_resource phmr;
  auto packed = cudf::pack(table_view, stream, phmr);
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto unpacked = cudf::unpack(packed); });
}

NVBENCH_BENCH(bench_host_unpack)
  .set_name("host_unpack")
  .add_int64_axis("num_rows", {4096, 32768, 262144})
  .add_int64_axis("num_cols", {64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});

static void bench_host_unpack_and_column_access(nvbench::state& state)
{
  const auto table      = make_bench_table(state);
  const auto table_view = table->view();
  auto stream           = cudf::get_default_stream();
  rmm::mr::pinned_host_memory_resource phmr;
  auto packed = cudf::pack(table_view, stream, phmr);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto unpacked = cudf::unpack(packed);
    column_sum(unpacked.column(0));
  });
}

NVBENCH_BENCH(bench_host_unpack_and_column_access)
  .set_name("host_unpack_and_column_access")
  .add_int64_axis("num_rows", {4096, 32768, 262144})
  .add_int64_axis("num_cols", {64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});
