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

#include <cudf/groupby.hpp>

#include <nvbench/nvbench.cuh>

template <typename Type>
void bench_groupby_max(nvbench::state& state, nvbench::type_list<Type>)
{
  cudf::rmm_pool_raii pool_raii;
  const auto size = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  auto const keys_table = [&] {
    data_profile profile;
    profile.set_null_frequency(std::nullopt);
    profile.set_cardinality(0);
    profile.set_distribution_params<int32_t>(
      cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, 100);
    return create_random_table({cudf::type_to_id<int32_t>()}, row_count{size}, profile);
  }();

  auto const vals_table = [&] {
    data_profile profile;
    if (const auto null_freq = state.get_float64("null_frequency"); null_freq > 0) {
      profile.set_null_frequency({null_freq});
    } else {
      profile.set_null_frequency(std::nullopt);
    }
    profile.set_cardinality(0);
    profile.set_distribution_params<Type>(cudf::type_to_id<Type>(),
                                          distribution_id::UNIFORM,
                                          static_cast<Type>(0),
                                          static_cast<Type>(1000));
    return create_random_table({cudf::type_to_id<Type>()}, row_count{size}, profile);
  }();

  auto const& keys = keys_table->get_column(0);
  auto const& vals = vals_table->get_column(0);

  auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys, keys, keys}));

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::default_stream_value.value()));
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto const result = gb_obj.aggregate(requests); });
}

NVBENCH_BENCH_TYPES(bench_groupby_max,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t, float, double>))
  .set_name("groupby_max")
  .add_int64_power_of_two_axis("num_rows", {12, 18, 24})
  .add_float64_axis("null_frequency", {0, 0.1, 0.9});
