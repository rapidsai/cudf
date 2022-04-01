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
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

class Groupby : public cudf::benchmark {
};

template <cudf::rank_method method>
void BM_group_rank(benchmark::State& state)
{
  using namespace cudf;

  const size_type column_size{(size_type)state.range(0)};
  const int num_groups = 100;

  data_profile profile;
  profile.set_null_frequency(std::nullopt);
  profile.set_cardinality(0);
  profile.set_distribution_params<int64_t>(
    cudf::type_to_id<int64_t>(), distribution_id::UNIFORM, 0, num_groups);

  auto source_table = create_random_table(
    {cudf::type_to_id<int64_t>(), cudf::type_to_id<int64_t>()}, row_count{column_size}, profile);

  // TODO values to be sorted too for groupby rank
  // auto sorted_table = cudf::sort(*source_table);

  auto agg = cudf::make_rank_aggregation<groupby_scan_aggregation>(method);
  std::vector<groupby::scan_request> requests;
  requests.emplace_back(groupby::scan_request());
  requests[0].values = source_table->view().column(1);
  requests[0].aggregations.push_back(std::move(agg));

  groupby::groupby gb_obj(
    table_view{{source_table->view().column(0)}}, null_policy::EXCLUDE, sorted::NO);

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    // groupby scan uses sort implementation
    auto result = gb_obj.scan(requests);
  }
}
//

BENCHMARK_DEFINE_F(Groupby, rank_dense)(::benchmark::State& state)
{
  BM_group_rank<cudf::rank_method::DENSE>(state);
}

BENCHMARK_REGISTER_F(Groupby, rank_dense)
  ->Arg(1'000'000)
  ->Arg(10'000'000)
  ->Arg(100'000'000)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(Groupby, rank_min)(::benchmark::State& state)
{
  BM_group_rank<cudf::rank_method::MIN>(state);
}

BENCHMARK_REGISTER_F(Groupby, rank_min)
  ->Arg(1'000'000)
  ->Arg(10'000'000)
  ->Arg(100'000'000)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(Groupby, rank_max)(::benchmark::State& state)
{
  BM_group_rank<cudf::rank_method::MAX>(state);
}

BENCHMARK_REGISTER_F(Groupby, rank_max)
  ->Arg(1'000'000)
  ->Arg(10'000'000)
  ->Arg(100'000'000)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(Groupby, rank_first)(::benchmark::State& state)
{
  BM_group_rank<cudf::rank_method::FIRST>(state);
}

BENCHMARK_REGISTER_F(Groupby, rank_first)
  ->Arg(1'000'000)
  ->Arg(10'000'000)
  ->Arg(100'000'000)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(Groupby, rank_average)(::benchmark::State& state)
{
  BM_group_rank<cudf::rank_method::AVERAGE>(state);
}

BENCHMARK_REGISTER_F(Groupby, rank_average)
  ->Arg(1'000'000)
  ->Arg(10'000'000)
  ->Arg(100'000'000)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);
