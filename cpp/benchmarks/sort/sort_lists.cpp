/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/rmm_pool_raii.hpp>

#include <cudf/detail/sorting.hpp>

#include <nvbench/nvbench.cuh>

void nvbench_sort_lists(nvbench::state& state)
{
  cudf::rmm_pool_raii pool_raii;

  const size_t size_bytes(state.get_int64("size_bytes"));
  const cudf::size_type depth{static_cast<cudf::size_type>(state.get_int64("depth"))};
  auto const null_frequency{state.get_float64("null_frequency")};

  data_profile table_profile;
  table_profile.set_distribution_params(cudf::type_id::LIST, distribution_id::UNIFORM, 0, 5);
  table_profile.set_list_depth(depth);
  table_profile.set_null_probability(null_frequency);
  auto const table =
    create_random_table({cudf::type_id::LIST}, table_size_bytes{size_bytes}, table_profile);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    cudf::detail::sorted_order(*table, {}, {}, stream_view, rmm::mr::get_current_device_resource());
  });
}

NVBENCH_BENCH(nvbench_sort_lists)
  .set_name("sort_list")
  .add_int64_power_of_two_axis("size_bytes", {10, 18, 24, 28})
  .add_int64_axis("depth", {1, 4})
  .add_float64_axis("null_frequency", {0, 0.2});
