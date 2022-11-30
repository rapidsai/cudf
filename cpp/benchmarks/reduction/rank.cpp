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

#include <cudf/detail/scan.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/list_view.hpp>

#include <nvbench/nvbench.cuh>

template <typename type>
static void nvbench_reduction_scan(nvbench::state& state, nvbench::type_list<type>)
{
  cudf::rmm_pool_raii pool_raii;

  auto const dtype = cudf::type_to_id<type>();

  double const null_probability = state.get_float64("null_probability");
  size_t const size             = state.get_int64("data_size");

  data_profile const profile = data_profile_builder()
                                 .null_probability(null_probability)
                                 .distribution(dtype, distribution_id::UNIFORM, 0, 5);

  auto const table = create_random_table({dtype}, table_size_bytes{size / 2}, profile);

  auto const new_tbl = cudf::repeat(table->view(), 2);
  cudf::column_view input(new_tbl->view().column(0));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    auto result = cudf::detail::inclusive_dense_rank_scan(
      input, stream_view, rmm::mr::get_current_device_resource());
  });
}

using data_type = nvbench::type_list<int32_t, cudf::list_view>;

NVBENCH_BENCH_TYPES(nvbench_reduction_scan, NVBENCH_TYPE_AXES(data_type))
  .set_name("rank_scan")
  .add_float64_axis("null_probability", {0, 0.1, 0.5, 0.9})
  .add_int64_axis("data_size",
                  {
                    10000,      // 10k
                    100000,     // 100k
                    1000000,    // 1M
                    10000000,   // 10M
                    100000000,  // 100M
                  });
