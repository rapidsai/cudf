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

#include <benchmarks/common/generate_nested_types.hpp>

#include <cudf/detail/merge.hpp>
#include <cudf/detail/sorting.hpp>

#include <nvbench/nvbench.cuh>

void nvbench_merge_struct(nvbench::state& state)
{
  rmm::cuda_stream_view stream;

  auto const input1 = create_structs_data(state);
  auto const sorted_input1 =
    cudf::detail::sort(*input1, {}, {}, stream, rmm::mr::get_current_device_resource());

  auto const input2 = create_structs_data(state);
  auto const sorted_input2 =
    cudf::detail::sort(*input2, {}, {}, stream, rmm::mr::get_current_device_resource());

  stream.synchronize();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};

    cudf::detail::merge({*sorted_input1, *sorted_input2},
                        {0},
                        {cudf::order::ASCENDING},
                        {},
                        stream_view,
                        rmm::mr::get_current_device_resource());
  });
}

NVBENCH_BENCH(nvbench_merge_struct)
  .set_name("merge_struct")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {0, 1, 8})
  .add_int64_axis("Nulls", {0, 1});
