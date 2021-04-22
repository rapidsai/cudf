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

class Rank : public cudf::benchmark {
};

static void BM_rank(benchmark::State& state, bool nulls)
{
  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  const cudf::size_type n_rows{(cudf::size_type)state.range(0)};

  // Create columns with values in the range [0,100)
  column_wrapper input = [&, n_rows]() {
    auto elements = cudf::detail::make_counting_transform_iterator(
      0, [&](auto row) { return distribution(generator); });
    if (!nulls) return column_wrapper(elements, elements + n_rows);
    auto valids = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return i % 100 == 0 ? false : true; });
    return column_wrapper(elements, elements + n_rows, valids);
  }();

  for (auto _ : state) {
    cuda_event_timer raii(state, true, rmm::cuda_stream_default);

    auto result = cudf::rank(input,
                             cudf::rank_method::FIRST,
                             cudf::order::ASCENDING,
                             nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE,
                             cudf::null_order::AFTER,
                             false);
  }
}

#define RANK_BENCHMARK_DEFINE(name, nulls)          \
  BENCHMARK_DEFINE_F(Rank, name)                    \
  (::benchmark::State & st) { BM_rank(st, nulls); } \
  BENCHMARK_REGISTER_F(Rank, name)                  \
    ->RangeMultiplier(8)                            \
    ->Ranges({{1 << 10, 1 << 26}})                  \
    ->UseManualTime()                               \
    ->Unit(benchmark::kMillisecond);

RANK_BENCHMARK_DEFINE(no_nulls, false)
RANK_BENCHMARK_DEFINE(nulls, true)
