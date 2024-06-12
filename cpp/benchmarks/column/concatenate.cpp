/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <benchmarks/fixture/templated_benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <vector>

class Concatenate : public cudf::benchmark {};

template <typename T, bool Nullable>
static void BM_concatenate(benchmark::State& state)
{
  cudf::size_type const num_rows = state.range(0);
  cudf::size_type const num_cols = state.range(1);

  auto input         = create_sequence_table(cycle_dtypes({cudf::type_to_id<T>()}, num_cols),
                                     row_count{num_rows},
                                     Nullable ? std::optional<double>{2.0 / 3.0} : std::nullopt);
  auto input_columns = input->view();
  std::vector<cudf::column_view> column_views(input_columns.begin(), input_columns.end());

  CUDF_CHECK_CUDA(0);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    auto result = cudf::concatenate(column_views);
  }

  state.SetBytesProcessed(state.iterations() * num_cols * num_rows * sizeof(T));
}

#define CONCAT_BENCHMARK_DEFINE(type, nullable)                             \
  BENCHMARK_DEFINE_F(Concatenate, BM_concatenate##_##nullable_##nullable)   \
  (::benchmark::State & st) { BM_concatenate<type, nullable>(st); }         \
  BENCHMARK_REGISTER_F(Concatenate, BM_concatenate##_##nullable_##nullable) \
    ->RangeMultiplier(8)                                                    \
    ->Ranges({{1 << 6, 1 << 18}, {2, 1024}})                                \
    ->Unit(benchmark::kMillisecond)                                         \
    ->UseManualTime();

CONCAT_BENCHMARK_DEFINE(int64_t, false)
CONCAT_BENCHMARK_DEFINE(int64_t, true)

template <typename T, bool Nullable>
static void BM_concatenate_tables(benchmark::State& state)
{
  cudf::size_type const num_rows   = state.range(0);
  cudf::size_type const num_cols   = state.range(1);
  cudf::size_type const num_tables = state.range(2);

  std::vector<std::unique_ptr<cudf::table>> tables(num_tables);
  std::generate_n(tables.begin(), num_tables, [&]() {
    return create_sequence_table(cycle_dtypes({cudf::type_to_id<T>()}, num_cols),
                                 row_count{num_rows},
                                 Nullable ? std::optional<double>{2.0 / 3.0} : std::nullopt);
  });

  // Generate table views
  std::vector<cudf::table_view> table_views(num_tables);
  std::transform(tables.begin(), tables.end(), table_views.begin(), [](auto& table) mutable {
    return table->view();
  });

  CUDF_CHECK_CUDA(0);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    auto result = cudf::concatenate(table_views);
  }

  state.SetBytesProcessed(state.iterations() * num_cols * num_rows * num_tables * sizeof(T));
}

#define CONCAT_TABLES_BENCHMARK_DEFINE(type, nullable)                             \
  BENCHMARK_DEFINE_F(Concatenate, BM_concatenate_tables##_##nullable_##nullable)   \
  (::benchmark::State & st) { BM_concatenate_tables<type, nullable>(st); }         \
  BENCHMARK_REGISTER_F(Concatenate, BM_concatenate_tables##_##nullable_##nullable) \
    ->RangeMultiplier(8)                                                           \
    ->Ranges({{1 << 8, 1 << 12}, {2, 32}, {2, 128}})                               \
    ->Unit(benchmark::kMillisecond)                                                \
    ->UseManualTime();

CONCAT_TABLES_BENCHMARK_DEFINE(int64_t, false)
CONCAT_TABLES_BENCHMARK_DEFINE(int64_t, true)

class ConcatenateStrings : public cudf::benchmark {};

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

  CUDF_CHECK_CUDA(0);

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    auto result = cudf::concatenate(column_views);
  }

  state.SetBytesProcessed(state.iterations() * num_cols * num_rows *
                          (sizeof(int32_t) + num_chars));  // offset + chars
}

#define CONCAT_STRINGS_BENCHMARK_DEFINE(nullable)                                   \
  BENCHMARK_DEFINE_F(Concatenate, BM_concatenate_strings##_##nullable_##nullable)   \
  (::benchmark::State & st) { BM_concatenate_strings<nullable>(st); }               \
  BENCHMARK_REGISTER_F(Concatenate, BM_concatenate_strings##_##nullable_##nullable) \
    ->RangeMultiplier(8)                                                            \
    ->Ranges({{1 << 8, 1 << 14}, {8, 128}, {2, 256}})                               \
    ->Unit(benchmark::kMillisecond)                                                 \
    ->UseManualTime();

CONCAT_STRINGS_BENCHMARK_DEFINE(false)
CONCAT_STRINGS_BENCHMARK_DEFINE(true)
