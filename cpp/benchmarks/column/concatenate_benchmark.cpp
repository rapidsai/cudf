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
#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>

#include <tests/utilities/column_wrapper.hpp>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <vector>

template <typename T, bool Nullable>
class Concatenate : public cudf::benchmark {
};

template <typename T, bool Nullable>
static void BM_concatenate(benchmark::State& state)
{
  using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

  auto const num_rows = state.range(0);
  auto const num_cols = state.range(1);

  // Create owning columns
  std::vector<column_wrapper> columns;
  columns.reserve(num_cols);
  std::generate_n(std::back_inserter(columns), num_cols, [num_rows]() {
    auto iter = thrust::make_counting_iterator(0);
    if (Nullable) {
      auto valid_iter = thrust::make_transform_iterator(iter, [](auto i) { return i % 3 == 0; });
      return column_wrapper(iter, iter + num_rows, valid_iter);
    } else {
      return column_wrapper(iter, iter + num_rows);
    }
  });

  // Generate column views
  std::vector<cudf::column_view> column_views;
  column_views.reserve(columns.size());
  std::transform(
    columns.begin(), columns.end(), std::back_inserter(column_views), [](auto const& col) {
      return static_cast<cudf::column_view>(col);
    });

  CHECK_CUDA(0);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    auto result = cudf::concatenate(column_views);
  }

  state.SetBytesProcessed(state.iterations() * num_cols * num_rows * sizeof(T));
}

#define CONCAT_BENCHMARK_DEFINE(name, type, nullable)                     \
  BENCHMARK_TEMPLATE_DEFINE_F(Concatenate, name, type, nullable)          \
  (::benchmark::State & state) { BM_concatenate<type, nullable>(state); } \
  BENCHMARK_REGISTER_F(Concatenate, name)                                 \
    ->RangeMultiplier(8)                                                  \
    ->Ranges({{1 << 6, 1 << 18}, {2, 1024}})                              \
    ->Unit(benchmark::kMillisecond)                                       \
    ->UseManualTime();

CONCAT_BENCHMARK_DEFINE(concat_columns_int64_non_null, int64_t, false)
CONCAT_BENCHMARK_DEFINE(concat_columns_int64_nullable, int64_t, true)

template <typename T, bool Nullable>
static void BM_concatenate_tables(benchmark::State& state)
{
  using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

  auto const num_rows   = state.range(0);
  auto const num_cols   = state.range(1);
  auto const num_tables = state.range(2);

  // Create owning columns
  std::vector<column_wrapper> columns;
  columns.reserve(num_cols);
  std::generate_n(std::back_inserter(columns), num_cols * num_tables, [num_rows]() {
    auto iter = thrust::make_counting_iterator(0);
    if (Nullable) {
      auto valid_iter = thrust::make_transform_iterator(iter, [](auto i) { return i % 3 == 0; });
      return column_wrapper(iter, iter + num_rows, valid_iter);
    } else {
      return column_wrapper(iter, iter + num_rows);
    }
  });

  // Generate column views
  std::vector<std::vector<cudf::column_view>> column_views(num_tables);
  for (int i = 0; i < num_tables; ++i) {
    column_views[i].reserve(num_cols);
    auto it = columns.begin() + (i * num_cols);
    std::transform(it, it + num_cols, std::back_inserter(column_views[i]), [](auto const& col) {
      return static_cast<cudf::column_view>(col);
    });
  }

  // Generate table views
  std::vector<cudf::table_view> table_views;
  table_views.reserve(num_tables);
  std::transform(column_views.begin(),
                 column_views.end(),
                 std::back_inserter(table_views),
                 [](auto const& col_vec) { return cudf::table_view(col_vec); });

  CHECK_CUDA(0);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    auto result = cudf::concatenate(table_views);
  }

  state.SetBytesProcessed(state.iterations() * num_cols * num_rows * num_tables * sizeof(T));
}

#define CONCAT_TABLES_BENCHMARK_DEFINE(name, type, nullable)                     \
  BENCHMARK_TEMPLATE_DEFINE_F(Concatenate, name, type, nullable)                 \
  (::benchmark::State & state) { BM_concatenate_tables<type, nullable>(state); } \
  BENCHMARK_REGISTER_F(Concatenate, name)                                        \
    ->RangeMultiplier(8)                                                         \
    ->Ranges({{1 << 8, 1 << 12}, {2, 32}, {2, 128}})                             \
    ->Unit(benchmark::kMillisecond)                                              \
    ->UseManualTime();

CONCAT_TABLES_BENCHMARK_DEFINE(concat_tables_int64_non_null, int64_t, false)
CONCAT_TABLES_BENCHMARK_DEFINE(concat_tables_int64_nullable, int64_t, true)

template <bool Nullable>
class ConcatenateStrings : public cudf::benchmark {
};

template <bool Nullable>
static void BM_concatenate_strings(benchmark::State& state)
{
  using column_wrapper = cudf::test::strings_column_wrapper;

  auto const num_rows  = state.range(0);
  auto const num_chars = state.range(1);
  auto const num_cols  = state.range(2);

  std::string str(num_chars, 'a');

  // Create owning columns
  std::vector<column_wrapper> columns;
  columns.reserve(num_cols);
  std::generate_n(std::back_inserter(columns), num_cols, [num_rows, c_str = str.c_str()]() {
    auto iter = thrust::make_constant_iterator(c_str);
    if (Nullable) {
      auto count_it = thrust::make_counting_iterator(0);
      auto valid_iter =
        thrust::make_transform_iterator(count_it, [](auto i) { return i % 3 == 0; });
      return column_wrapper(iter, iter + num_rows, valid_iter);
    } else {
      return column_wrapper(iter, iter + num_rows);
    }
  });

  // Generate column views
  std::vector<cudf::column_view> column_views;
  column_views.reserve(columns.size());
  std::transform(
    columns.begin(), columns.end(), std::back_inserter(column_views), [](auto const& col) {
      return static_cast<cudf::column_view>(col);
    });

  CHECK_CUDA(0);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);
    auto result = cudf::concatenate(column_views);
  }

  state.SetBytesProcessed(state.iterations() * num_cols * num_rows *
                          (sizeof(int32_t) + num_chars));  // offset + chars
}

#define CONCAT_STRINGS_BENCHMARK_DEFINE(name, nullable)                     \
  BENCHMARK_TEMPLATE_DEFINE_F(ConcatenateStrings, name, nullable)           \
  (::benchmark::State & state) { BM_concatenate_strings<nullable>(state); } \
  BENCHMARK_REGISTER_F(ConcatenateStrings, name)                            \
    ->RangeMultiplier(8)                                                    \
    ->Ranges({{1 << 8, 1 << 14}, {8, 128}, {2, 256}})                       \
    ->Unit(benchmark::kMillisecond)                                         \
    ->UseManualTime();

CONCAT_STRINGS_BENCHMARK_DEFINE(concat_string_columns_non_null, false)
CONCAT_STRINGS_BENCHMARK_DEFINE(concat_string_columns_nullable, true)
