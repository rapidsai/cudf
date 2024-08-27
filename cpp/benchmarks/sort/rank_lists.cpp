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

#include <cudf_test/column_utilities.hpp>

#include <cudf/sorting.hpp>

#include <nvbench/nvbench.cuh>

template <cudf::rank_method method>
void nvbench_rank_lists(nvbench::state& state, nvbench::type_list<nvbench::enum_type<method>>)
{
  auto const table = create_lists_data(state);

  auto const null_frequency{state.get_float64("null_frequency")};

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::rank(table->view().column(0),
               method,
               cudf::order::ASCENDING,
               null_frequency ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE,
               cudf::null_order::AFTER,
               true,
               cudf::get_default_stream(),
               rmm::mr::get_current_device_resource());
  });
}

NVBENCH_BENCH_TYPES(nvbench_rank_lists, NVBENCH_TYPE_AXES(methods))
  .set_name("rank_lists")
  .add_int64_power_of_two_axis("size_bytes", {10, 18, 24, 28})
  .add_int64_axis("depth", {1, 4})
  .add_float64_axis("null_frequency", {0, 0.2});
