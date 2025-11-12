/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_nested_types.hpp>

#include <nvbench/nvbench.cuh>

void nvbench_create_structs(nvbench::state& state)
{
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto const table_ptr = create_structs_data(state); });
}

NVBENCH_BENCH(nvbench_create_structs)
  .set_name("create_structs")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {1, 8, 16})
  .add_int64_axis("Nulls", {0, 1});
