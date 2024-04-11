/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/groupby/group_common.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>

class Groupby : public cudf::benchmark {};

void BM_basic_sum(benchmark::State& state)
{
  cudf::size_type const column_size{(cudf::size_type)state.range(0)};

  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<int64_t>(), distribution_id::UNIFORM, 0, 100);
  auto keys = create_random_column(cudf::type_to_id<int64_t>(), row_count{column_size}, profile);
  auto vals = create_random_column(cudf::type_to_id<int64_t>(), row_count{column_size}, profile);

  cudf::groupby::groupby gb_obj(cudf::table_view({keys->view(), keys->view(), keys->view()}));

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  for (auto _ : state) {
    cuda_event_timer timer(state, true);

    auto result = gb_obj.aggregate(requests);
  }
}

BENCHMARK_DEFINE_F(Groupby, Basic)(::benchmark::State& state) { BM_basic_sum(state); }

BENCHMARK_REGISTER_F(Groupby, Basic)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Arg(10000)
  ->Arg(1000000)
  ->Arg(10000000)
  ->Arg(100000000);

void BM_pre_sorted_sum(benchmark::State& state)
{
  cudf::size_type const column_size{(cudf::size_type)state.range(0)};

  data_profile profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<int64_t>(), distribution_id::UNIFORM, 0, 100);
  auto keys_table =
    create_random_table({cudf::type_to_id<int64_t>()}, row_count{column_size}, profile);
  profile.set_null_probability(0.1);
  auto vals = create_random_column(cudf::type_to_id<int64_t>(), row_count{column_size}, profile);

  auto sort_order  = cudf::sorted_order(*keys_table);
  auto sorted_keys = cudf::gather(*keys_table, *sort_order);
  // No need to sort values using sort_order because they were generated randomly

  cudf::groupby::groupby gb_obj(*sorted_keys, cudf::null_policy::EXCLUDE, cudf::sorted::YES);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  for (auto _ : state) {
    cuda_event_timer timer(state, true);

    auto result = gb_obj.aggregate(requests);
  }
}

BENCHMARK_DEFINE_F(Groupby, PreSorted)(::benchmark::State& state) { BM_pre_sorted_sum(state); }

BENCHMARK_REGISTER_F(Groupby, PreSorted)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Arg(1000000)
  ->Arg(10000000)
  ->Arg(100000000);
