/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_nested_types.hpp>

#include <cudf/sorting.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

void nvbench_sort_struct(nvbench::state& state)
{
  auto const input = create_structs_data(state);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    cudf::sorted_order(*input, {}, {}, stream_view, cudf::get_current_device_resource_ref());
  });
}

NVBENCH_BENCH(nvbench_sort_struct)
  .set_name("sort_struct")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});
