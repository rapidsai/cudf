/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/structs/structs_column_view.hpp>

static constexpr cudf::size_type num_struct_members = 8;
static constexpr cudf::size_type max_int            = 100;
static constexpr cudf::size_type max_str_length     = 32;

static auto create_data_table(cudf::size_type n_rows)
{
  data_profile const table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, max_int)
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);

  // The first two struct members are int32 and string.
  // The first column is also used as keys in groupby.
  // The subsequent struct members are int32 and string again.
  return create_random_table(
    cycle_dtypes({cudf::type_id::INT32, cudf::type_id::STRING}, num_struct_members),
    row_count{n_rows},
    table_profile);
}

// Max aggregation/scan technically has the same performance as min.
template <typename OpType>
void BM_groupby_min_struct(benchmark::State& state)
{
  auto const n_rows = static_cast<cudf::size_type>(state.range(0));
  auto data_cols    = create_data_table(n_rows)->release();

  auto const keys_view = data_cols.front()->view();
  auto const values =
    cudf::make_structs_column(keys_view.size(), std::move(data_cols), 0, rmm::device_buffer());

  using RequestType = std::conditional_t<std::is_same_v<OpType, cudf::groupby_aggregation>,
                                         cudf::groupby::aggregation_request,
                                         cudf::groupby::scan_request>;

  auto gb_obj   = cudf::groupby::groupby(cudf::table_view({keys_view}));
  auto requests = std::vector<RequestType>();
  requests.emplace_back(RequestType());
  requests.front().values = values->view();
  requests.front().aggregations.push_back(cudf::make_min_aggregation<OpType>());

  for (auto _ : state) {
    [[maybe_unused]] auto const timer = cuda_event_timer(state, true);
    if constexpr (std::is_same_v<OpType, cudf::groupby_aggregation>) {
      [[maybe_unused]] auto const result = gb_obj.aggregate(requests);
    } else {
      [[maybe_unused]] auto const result = gb_obj.scan(requests);
    }
  }
}

class Groupby : public cudf::benchmark {};

#define MIN_RANGE 10'000
#define MAX_RANGE 10'000'000

#define REGISTER_BENCHMARK(name, op_type)                       \
  BENCHMARK_DEFINE_F(Groupby, name)(::benchmark::State & state) \
  {                                                             \
    BM_groupby_min_struct<op_type>(state);                      \
  }                                                             \
  BENCHMARK_REGISTER_F(Groupby, name)                           \
    ->UseManualTime()                                           \
    ->Unit(benchmark::kMillisecond)                             \
    ->RangeMultiplier(4)                                        \
    ->Ranges({{MIN_RANGE, MAX_RANGE}});

REGISTER_BENCHMARK(Aggregation, cudf::groupby_aggregation)
REGISTER_BENCHMARK(Scan, cudf::groupby_scan_aggregation)
