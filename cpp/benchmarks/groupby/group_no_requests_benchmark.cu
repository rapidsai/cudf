/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>

#include <memory>
#include <random>

class Groupby : public cudf::benchmark {
};

// TODO: put it in a struct so `uniform` can be remade with different min, max
template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

void BM_basic_no_requests(benchmark::State& state)
{
  using wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

  const cudf::size_type column_size{(cudf::size_type)state.range(0)};

  auto data_it = cudf::detail::make_counting_transform_iterator(
    0, [=](cudf::size_type row) { return random_int(0, 100); });

  wrapper keys(data_it, data_it + column_size);
  wrapper vals(data_it, data_it + column_size);

  std::vector<cudf::groupby::aggregation_request> requests;

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    cudf::groupby::groupby gb_obj(cudf::table_view({keys}));
    auto result = gb_obj.aggregate(requests);
  }
}

BENCHMARK_DEFINE_F(Groupby, BasicNoRequest)(::benchmark::State& state)
{
  BM_basic_no_requests(state);
}

BENCHMARK_REGISTER_F(Groupby, BasicNoRequest)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Arg(10000)
  ->Arg(1000000)
  ->Arg(10000000)
  ->Arg(100000000);

void BM_pre_sorted_no_requests(benchmark::State& state)
{
  using wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

  const cudf::size_type column_size{(cudf::size_type)state.range(0)};

  auto data_it = cudf::detail::make_counting_transform_iterator(
    0, [=](cudf::size_type row) { return random_int(0, 100); });
  auto valid_it = cudf::detail::make_counting_transform_iterator(
    0, [=](cudf::size_type row) { return random_int(0, 100) < 90; });

  wrapper keys(data_it, data_it + column_size);
  wrapper vals(data_it, data_it + column_size, valid_it);

  auto keys_table  = cudf::table_view({keys});
  auto sort_order  = cudf::sorted_order(keys_table);
  auto sorted_keys = cudf::gather(keys_table, *sort_order);
  // No need to sort values using sort_order because they were generated randomly

  std::vector<cudf::groupby::aggregation_request> requests;

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    cudf::groupby::groupby gb_obj(*sorted_keys, cudf::null_policy::EXCLUDE, cudf::sorted::YES);
    auto result = gb_obj.aggregate(requests);
  }
}

BENCHMARK_DEFINE_F(Groupby, PreSortedNoRequests)(::benchmark::State& state)
{
  BM_pre_sorted_no_requests(state);
}

BENCHMARK_REGISTER_F(Groupby, PreSortedNoRequests)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Arg(1000000)
  ->Arg(10000000)
  ->Arg(100000000);
