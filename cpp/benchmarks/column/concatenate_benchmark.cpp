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

#include <cudf/column/column.hpp>

#include <tests/utilities/column_wrapper.hpp>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <vector>
#include <algorithm>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
class Concatenate : public cudf::benchmark {};

template<typename T>
static void BM_concatenate(benchmark::State& state) {
  auto const num_cols = state.range(0);
  auto const num_rows = state.range(1);

  // Create owning columns
  std::vector<column_wrapper<T>> columns;
  columns.reserve(num_cols);
  std::generate_n(std::back_inserter(columns), num_cols,
    [num_rows]() {
      auto iter = thrust::make_counting_iterator(0);
      return column_wrapper<T>(iter, iter + num_rows);
    });

  // Generate column views
  std::vector<cudf::column_view> column_views;
  column_views.reserve(num_cols);
  std::transform(columns.begin(), columns.end(),
    std::back_inserter(column_views),
    [](auto const& col) {
      return static_cast<cudf::column_view>(col);
    });

  CHECK_CUDA(0);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    auto result = cudf::concatenate(column_views);
  }

  state.SetBytesProcessed(state.iterations() * num_cols * num_rows * sizeof(T));
}

#define CONCAT_BENCHMARK_DEFINE(name, type)           \
BENCHMARK_TEMPLATE_DEFINE_F(Concatenate, name, type)  \
(::benchmark::State& state) {                         \
  BM_concatenate<type>(state);                        \
}                                                     \
BENCHMARK_REGISTER_F(Concatenate, name)               \
  ->RangeMultiplier(4)                                \
  ->Ranges({{2, 1024}, {1<<6, 1<<18}})                 \
  ->Unit(benchmark::kMillisecond)                     \
  ->UseManualTime();

CONCAT_BENCHMARK_DEFINE(concat_columns_int64, int64_t)
