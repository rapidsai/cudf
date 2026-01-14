/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/copying.hpp>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reverse.h>
#include <thrust/shuffle.h>

#include <nvbench/nvbench.cuh>

static void bench_scatter(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const coalesce = static_cast<bool>(state.get_int64("coalesce"));

  // Gather indices
  auto scatter_map_table =
    create_sequence_table({cudf::type_to_id<cudf::size_type>()}, row_count{num_rows});
  auto scatter_map = scatter_map_table->get_column(0).mutable_view();

  if (coalesce) {
    thrust::reverse(
      thrust::device, scatter_map.begin<cudf::size_type>(), scatter_map.end<cudf::size_type>());
  } else {
    thrust::shuffle(thrust::device,
                    scatter_map.begin<cudf::size_type>(),
                    scatter_map.end<cudf::size_type>(),
                    thrust::default_random_engine());
  }

  // Every element is valid
  auto source_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols),
                                            row_count{num_rows});
  auto target_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols),
                                            row_count{num_rows});

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int8_t>(source_table->alloc_size());
  state.add_global_memory_writes<int8_t>(target_table->alloc_size());

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::scatter(*source_table, scatter_map, *target_table); });
}

NVBENCH_BENCH(bench_scatter)
  .set_name("scatter")
  .add_int64_axis("num_rows", {64, 512, 4096, 32768, 262144, 2097152, 16777216, 134217728})
  .add_int64_axis("num_cols", {1, 8})
  .add_int64_axis("coalesce", {true, false});
