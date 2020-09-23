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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/merge.hpp>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <random>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

// Fixture that enables RMM pool mode
class Merge : public cudf::benchmark {
};

using IntColWrap = cudf::test::fixed_width_column_wrapper<int32_t>;

void BM_merge(benchmark::State& state)
{
  cudf::size_type const avg_rows = 1 << 19;  // 512K rows
  int const num_tables           = state.range(0);

  // Content is irrelevant for the benchmark
  auto data_sequence = thrust::make_constant_iterator(0);

  // Using 0 seed to ensure consistent pseudo-numbers on each run
  std::mt19937 rand_gen(0);
  // Gaussian distribution with 98% of elements are in range [0, avg_rows*2]
  std::normal_distribution<> table_size_dist(avg_rows, avg_rows / 2);
  // Used to generate a random monotonic sequence for each table key column
  std::uniform_int_distribution<> key_dist(0, 10);

  std::vector<std::pair<IntColWrap, IntColWrap>> columns;
  size_t total_rows = 0;
  std::vector<cudf::table_view> tables;
  for (int i = 0; i < num_tables; ++i) {
    cudf::size_type const rows = std::round(table_size_dist(rand_gen));
    // Ensure size in range [0, avg_rows*2]
    auto const clamped_rows = std::max(std::min(rows, avg_rows * 2), 0);

    int32_t prev_key  = 0;
    auto key_sequence = cudf::test::make_counting_transform_iterator(0, [&](auto row) {
      prev_key += key_dist(rand_gen);
      return prev_key;
    });

    columns.emplace_back(
      std::pair<IntColWrap, IntColWrap>{IntColWrap(key_sequence, key_sequence + clamped_rows),
                                        IntColWrap(data_sequence, data_sequence + clamped_rows)});
    tables.push_back(cudf::table_view{{columns.back().first, columns.back().second}});
    total_rows += clamped_rows;
  }
  std::vector<cudf::size_type> const key_cols{0};
  std::vector<cudf::order> const column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> const null_precedence{};

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    auto result = cudf::merge(tables, key_cols, column_order, null_precedence);
  }

  state.SetBytesProcessed(state.iterations() * 2 * sizeof(int32_t) * total_rows);
}

#define MBM_BENCHMARK_DEFINE(name)                                                 \
  BENCHMARK_DEFINE_F(Merge, name)(::benchmark::State & state) { BM_merge(state); } \
  BENCHMARK_REGISTER_F(Merge, name)                                                \
    ->Unit(benchmark::kNanosecond)                                                 \
    ->UseManualTime()                                                              \
    ->RangeMultiplier(2)                                                           \
    ->Ranges({{2, 128}});

MBM_BENCHMARK_DEFINE(pow2tables);
