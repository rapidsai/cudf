/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_nested_types.hpp>

#include <cudf/merge.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

void nvbench_merge_struct(nvbench::state& state)
{
  rmm::cuda_stream_view stream;

  auto const input1 = create_structs_data(state);
  auto const sorted_input1 =
    cudf::sort(*input1, {}, {}, stream, cudf::get_current_device_resource_ref());

  auto const input2 = create_structs_data(state);
  auto const sorted_input2 =
    cudf::sort(*input2, {}, {}, stream, cudf::get_current_device_resource_ref());

  stream.synchronize();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};

    cudf::merge({*sorted_input1, *sorted_input2},
                {0},
                {cudf::order::ASCENDING},
                {},
                stream_view,
                cudf::get_current_device_resource_ref());
  });
}

NVBENCH_BENCH(nvbench_merge_struct)
  .set_name("merge_struct")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});
