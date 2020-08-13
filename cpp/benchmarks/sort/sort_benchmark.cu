/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <benchmark/benchmark.h>

#include <cudf/sorting.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/table_utilities.hpp>

#include <cudf/types.hpp>

#include "../common/generate_benchmark_input.hpp"
#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

template <bool stable>
class Sort : public cudf::benchmark {
};

template <bool stable>
static void BM_sort(benchmark::State& state)
{
  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  const cudf::size_type n_rows{(cudf::size_type)state.range(0)};
  const cudf::size_type n_cols{(cudf::size_type)state.range(1)};
  auto type_size = cudf::size_of(cudf::data_type(cudf::type_to_id<Type>()));

  // Create columns with values in the range [0,100)
  std::vector<column_wrapper> columns;
  columns.reserve(n_cols);
  std::generate_n(std::back_inserter(columns), n_cols, [&, n_rows]() {
    auto valids = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return i % 100 == 0 ? false : true; });
    auto elements = cudf::test::make_counting_transform_iterator(
      0, [&](auto row) { return distribution(generator); });
    return column_wrapper(elements, elements + n_rows, valids);
  });

  // Create column views
  auto column_views = std::vector<cudf::column_view>(columns.begin(), columns.end());

  // Create table view
  auto input = cudf::table_view(column_views);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);

    auto result = (stable) ? cudf::stable_sorted_order(input) : cudf::sorted_order(input);
  }
}

#define SORT_BENCHMARK_DEFINE(name, stable)          \
  BENCHMARK_TEMPLATE_DEFINE_F(Sort, name, stable)    \
  (::benchmark::State & st) { BM_sort<stable>(st); } \
  BENCHMARK_REGISTER_F(Sort, name)                   \
    ->RangeMultiplier(8)                             \
    ->Ranges({{1 << 10, 1 << 26}, {1, 8}})           \
    ->UseManualTime()                                \
    ->Unit(benchmark::kMillisecond);

SORT_BENCHMARK_DEFINE(sort_stable, true)
SORT_BENCHMARK_DEFINE(sort_unstable, false)
