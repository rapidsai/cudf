/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "rank_types_common.hpp"

#include <benchmarks/common/generate_nested_types.hpp>

#include <cudf/sorting.hpp>

#include <nvbench/nvbench.cuh>

template <cudf::rank_method method>
void nvbench_rank_structs(nvbench::state& state, nvbench::type_list<nvbench::enum_type<method>>)
{
  auto const table = create_structs_data(state);

  bool const nulls{static_cast<bool>(state.get_int64("Nulls"))};

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::rank(table->view().column(0),
               method,
               cudf::order::ASCENDING,
               nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE,
               cudf::null_order::AFTER,
               rmm::mr::get_current_device_resource());
  });
}

NVBENCH_BENCH_TYPES(nvbench_rank_structs, NVBENCH_TYPE_AXES(methods))
  .set_name("rank_structs")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});
