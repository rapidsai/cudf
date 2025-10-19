/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <nvbench/nvbench.cuh>

static void bench_groupby_sum_scan(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<int64_t>(), distribution_id::UNIFORM, 0, 100);
  auto keys = create_random_column(cudf::type_to_id<int64_t>(), row_count{num_rows}, profile);
  auto vals = create_random_column(cudf::type_to_id<int64_t>(), row_count{num_rows}, profile);

  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back(cudf::groupby::scan_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::groupby::groupby gb_obj(cudf::table_view({keys->view(), keys->view(), keys->view()}));
    auto result = gb_obj.scan(requests);
  });
}

NVBENCH_BENCH(bench_groupby_sum_scan)
  .set_name("sum_scan")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000});

static void bench_groupby_pre_sorted_sum_scan(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  data_profile profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<int64_t>(), distribution_id::UNIFORM, 0, 100);
  auto keys_table =
    create_random_table({cudf::type_to_id<int64_t>()}, row_count{num_rows}, profile);
  profile.set_null_probability(0.1);
  auto vals = create_random_column(cudf::type_to_id<int64_t>(), row_count{num_rows}, profile);

  auto sort_order  = cudf::sorted_order(*keys_table);
  auto sorted_keys = cudf::gather(*keys_table, *sort_order);
  // No need to sort values using sort_order because they were generated randomly

  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back(cudf::groupby::scan_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::groupby::groupby gb_obj(*sorted_keys, cudf::null_policy::EXCLUDE, cudf::sorted::YES);
    auto result = gb_obj.scan(requests);
  });
}

NVBENCH_BENCH(bench_groupby_pre_sorted_sum_scan)
  .set_name("pre_sorted_sum_scan")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000});
