/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/sorting.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

template <bool stable>
class Sort : public cudf::benchmark {
};

template <bool stable>
static void BM_sort(benchmark::State& state, bool nulls)
{
  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  const cudf::size_type n_rows{(cudf::size_type)state.range(0)};
  const cudf::size_type n_cols{(cudf::size_type)state.range(1)};

  // Create columns with values in the range [0,100)
  std::vector<column_wrapper> columns;
  columns.reserve(n_cols);
  std::generate_n(std::back_inserter(columns), n_cols, [&, n_rows]() {
    auto elements = cudf::detail::make_counting_transform_iterator(
      0, [&](auto row) { return distribution(generator); });
    if (!nulls) return column_wrapper(elements, elements + n_rows);
    auto valids = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return i % 100 == 0 ? false : true; });
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

#define SORT_BENCHMARK_DEFINE(name, stable, nulls)          \
  BENCHMARK_TEMPLATE_DEFINE_F(Sort, name, stable)           \
  (::benchmark::State & st) { BM_sort<stable>(st, nulls); } \
  BENCHMARK_REGISTER_F(Sort, name)                          \
    ->RangeMultiplier(8)                                    \
    ->Ranges({{1 << 10, 1 << 26}, {1, 8}})                  \
    ->UseManualTime()                                       \
    ->Unit(benchmark::kMillisecond);

SORT_BENCHMARK_DEFINE(unstable_no_nulls, false, false)
SORT_BENCHMARK_DEFINE(stable_no_nulls, true, false)
SORT_BENCHMARK_DEFINE(unstable, false, true)
SORT_BENCHMARK_DEFINE(stable, true, true)
