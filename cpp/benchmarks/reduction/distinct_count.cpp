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

#include <cudf/detail/stream_compaction.hpp>

#include <nvbench/nvbench.cuh>

template <typename Type>
static void bench_reduction_distinct_count(nvbench::state& state, nvbench::type_list<Type>)
{
  cudf::rmm_pool_raii pool_raii;

  auto const dtype            = cudf::type_to_id<Type>();
  auto const size             = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const null_probability = state.get_float64("null_probability");

  data_profile profile;
  profile.set_distribution_params(dtype, distribution_id::UNIFORM, 0, size / 100);
  if (null_probability > 0) {
    profile.set_null_probability({null_probability});
  } else {
    profile.set_null_probability(std::nullopt);
  }

  auto const data_table   = create_random_table({dtype}, row_count{size}, profile);
  auto const& data_column = data_table->get_column(0);
  auto const input_table  = cudf::table_view{{data_column, data_column, data_column}};

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream{launch.get_stream()};
    cudf::detail::distinct_count(input_table, cudf::null_equality::EQUAL, stream);
  });
}

using data_type = nvbench::type_list<int32_t, int64_t, float, double>;

NVBENCH_BENCH_TYPES(bench_reduction_distinct_count, NVBENCH_TYPE_AXES(data_type))
  .set_name("reduction_distinct_count")
  .add_int64_axis("num_rows",
                  {
                    10000,      // 10k
                    100000,     // 100k
                    1000000,    // 1M
                    10000000,   // 10M
                    100000000,  // 100M
                  })
  .add_float64_axis("null_probability", {0, 0.5});
