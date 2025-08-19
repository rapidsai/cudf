/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cudf/quantiles.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

#include <nvbench/nvbench.cuh>

static void bench_quantiles(nvbench::state& state)
{
  cudf::size_type const num_rows{static_cast<cudf::size_type>(state.get_int64("num_rows"))};
  cudf::size_type const num_cols{static_cast<cudf::size_type>(state.get_int64("num_cols"))};
  cudf::size_type const num_quantiles{
    static_cast<cudf::size_type>(state.get_int64("num_quantiles"))};
  bool const nulls{static_cast<bool>(state.get_int64("nulls"))};

  auto const data_type = cudf::type_to_id<int32_t>();

  // Create columns with values in the range [0,100)
  data_profile profile =
    data_profile_builder().cardinality(0).distribution(data_type, distribution_id::UNIFORM, 0, 100);
  profile.set_null_probability(nulls ? std::optional{0.01}
                                     : std::nullopt);  // 1% nulls or no null mask (<0)

  auto input_table =
    create_random_table(cycle_dtypes({data_type}, num_cols), row_count{num_rows}, profile);
  auto input = cudf::table_view(*input_table);

  std::vector<double> q(num_quantiles);
  thrust::tabulate(thrust::seq, q.begin(), q.end(), [num_quantiles](auto i) {
    return i * (1.0f / num_quantiles);
  });

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::quantiles(input, q); });
}

NVBENCH_BENCH(bench_quantiles)
  .set_name("quantiles")
  .add_int64_power_of_two_axis("num_rows", {16, 18, 20, 22, 24, 26})
  .add_int64_axis("num_cols", {1, 2, 4, 8})
  .add_int64_axis("num_quantiles", {1, 4, 8, 12})
  .add_int64_axis("nulls", {0, 1});
