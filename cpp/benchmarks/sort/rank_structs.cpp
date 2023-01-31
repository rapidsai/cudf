/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "nested_types_common.hpp"

#include <cudf/sorting.hpp>

#include <nvbench/nvbench.cuh>

void nvbench_rank_structs(nvbench::state& state, cudf::rank_method method)
{
  cudf::rmm_pool_raii pool_raii;

  auto const table = create_structs_data(state);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::rank(table->view().column(0),
               method,
               cudf::order::ASCENDING,
               cudf::null_policy::INCLUDE,
               cudf::null_order::BEFORE,
               rmm::mr::get_current_device_resource());
  });
}

void nvbench_rank_structs_first(nvbench::state& state)
{
  nvbench_rank_structs(state, cudf::rank_method::FIRST);
}

void nvbench_rank_structs_dense(nvbench::state& state)
{
  nvbench_rank_structs(state, cudf::rank_method::DENSE);
}

void nvbench_rank_structs_min(nvbench::state& state)
{
  nvbench_rank_structs(state, cudf::rank_method::MIN);
}

void nvbench_rank_structs_average(nvbench::state& state)
{
  nvbench_rank_structs(state, cudf::rank_method::AVERAGE);
}

NVBENCH_BENCH(nvbench_rank_structs_first)
  .set_name("rank_structs_first")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});

NVBENCH_BENCH(nvbench_rank_structs_dense)
  .set_name("rank_structs_dense")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});

NVBENCH_BENCH(nvbench_rank_structs_min)
  .set_name("rank_structs_min")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});

NVBENCH_BENCH(nvbench_rank_structs_average)
  .set_name("rank_structs_average")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});
