/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

namespace {
template <typename Type>
std::unique_ptr<cudf::table> create_table_data(cudf::size_type n_rows,
                                               cudf::size_type n_cols,
                                               bool has_nulls = false)
{
  data_profile profile;
  profile.set_cardinality(0);
  profile.set_null_frequency(has_nulls ? 0.1 : 0.0);
  profile.set_distribution_params<Type>(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, Type{0}, Type{1000});

  // Deterministic benchmark, using the same starting seed value for each benchmark execution.
  static unsigned seed = 0;

  return create_random_table(
    cycle_dtypes({cudf::type_to_id<Type>()}, n_cols), row_count{n_rows}, profile, seed++);
}

template <typename Type>
std::unique_ptr<cudf::column> create_column_data(cudf::size_type n_rows, bool has_nulls = false)
{
  return std::move(create_table_data<Type>(n_rows, 1, has_nulls)->release().front());
}

}  // namespace

// -------------------------------------------------------------------------------------------------
class BinarySearch : public cudf::benchmark {
};

void BM_column(benchmark::State& state, bool nulls)
{
  auto const column_size{static_cast<cudf::size_type>(state.range(0))};
  auto const values_size = column_size;

  auto const init_data = cudf::make_fixed_width_scalar<float>(static_cast<float>(0));
  auto const column    = cudf::sequence(column_size, *init_data);
  if (nulls) {
    auto [column_null_mask, column_null_count] = create_random_null_mask(column->size(), 0.1, 1);
    column->set_null_mask(std::move(column_null_mask), column_null_count);
  }
  auto const values = create_column_data<float>(values_size, nulls);

  // column is already sorted.
  auto const data_table = cudf::table_view({*column});

  for ([[maybe_unused]] auto _ : state) {
    [[maybe_unused]] auto const timer = cuda_event_timer(state, true);
    [[maybe_unused]] auto const col   = cudf::upper_bound(data_table,
                                                        cudf::table_view({*values}),
                                                        {cudf::order::ASCENDING},
                                                        {cudf::null_order::BEFORE});
  }
}

#define BINARY_SEARCH_BENCHMARK_DEFINE(name, nulls)   \
  BENCHMARK_DEFINE_F(BinarySearch, name)              \
  (::benchmark::State & st) { BM_column(st, nulls); } \
  BENCHMARK_REGISTER_F(BinarySearch, name)            \
    ->UseManualTime()                                 \
    ->Unit(benchmark::kMillisecond)                   \
    ->Ranges({{100000, 100000000}});

BINARY_SEARCH_BENCHMARK_DEFINE(Column_AllValid, false)
BINARY_SEARCH_BENCHMARK_DEFINE(Column_HasNulls, true)

// -------------------------------------------------------------------------------------------------
void BM_table(benchmark::State& state)
{
  using Type = float;

  auto const num_columns{static_cast<cudf::size_type>(state.range(0))};
  auto const column_size{static_cast<cudf::size_type>(state.range(1))};
  auto const values_size = column_size;

  auto const data_table   = create_table_data<Type>(column_size, num_columns, true);
  auto const values_table = create_table_data<Type>(values_size, num_columns, true);

  auto const orders      = std::vector<cudf::order>(num_columns, cudf::order::ASCENDING);
  auto const null_orders = std::vector<cudf::null_order>(num_columns, cudf::null_order::BEFORE);
  auto const sorted      = cudf::sort(*data_table);

  for ([[maybe_unused]] auto _ : state) {
    [[maybe_unused]] auto const timer = cuda_event_timer(state, true);
    [[maybe_unused]] auto const col =
      cudf::lower_bound(sorted->view(), *values_table, orders, null_orders);
  }
}

BENCHMARK_DEFINE_F(BinarySearch, Table)(::benchmark::State& state) { BM_table(state); }

static void CustomArguments(benchmark::internal::Benchmark* b)
{
  for (int num_cols = 1; num_cols <= 10; num_cols *= 2)
    for (int col_size = 100000; col_size <= 100000000; col_size *= 10)
      b->Args({num_cols, col_size});
}

BENCHMARK_REGISTER_F(BinarySearch, Table)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Apply(CustomArguments);

// -------------------------------------------------------------------------------------------------
class Contains : public cudf::benchmark {
};

void BM_contains_scalar(benchmark::State& state, bool nulls)
{
  auto const column_size{static_cast<cudf::size_type>(state.range(0))};

  auto const column = create_column_data<cudf::size_type>(column_size, nulls);
  auto const value  = cudf::make_fixed_width_scalar<cudf::size_type>(column_size / 2);

  for ([[maybe_unused]] auto _ : state) {
    [[maybe_unused]] auto const timer  = cuda_event_timer(state, true);
    [[maybe_unused]] auto const result = cudf::contains(*column, *value);
  }
}

#define CONTAINS_SCALAR_BENCHMARK_DEFINE(name, nulls)          \
  BENCHMARK_DEFINE_F(Contains, name)                           \
  (::benchmark::State & st) { BM_contains_scalar(st, nulls); } \
  BENCHMARK_REGISTER_F(Contains, name)                         \
    ->RangeMultiplier(8)                                       \
    ->Ranges({{1 << 15, 1 << 28}})                             \
    ->UseManualTime()                                          \
    ->Unit(benchmark::kMillisecond);

CONTAINS_SCALAR_BENCHMARK_DEFINE(SearchScalar_AllValid, false)
CONTAINS_SCALAR_BENCHMARK_DEFINE(SearchScalar_Nulls, true)

// -------------------------------------------------------------------------------------------------
void BM_contains_column(benchmark::State& state, bool nulls)
{
  auto const column_size{static_cast<cudf::size_type>(state.range(0))};

  auto const table_data = create_table_data<cudf::size_type>(column_size, 2, nulls);

  for ([[maybe_unused]] auto _ : state) {
    [[maybe_unused]] auto const timer = cuda_event_timer(state, true);
    [[maybe_unused]] auto const result =
      cudf::contains(table_data->get_column(0).view(), table_data->get_column(1).view());
  }
}

#define CONTAINS_COLUMN_BENCHMARK_DEFINE(name, nulls)          \
  BENCHMARK_DEFINE_F(Contains, name)                           \
  (::benchmark::State & st) { BM_contains_column(st, nulls); } \
  BENCHMARK_REGISTER_F(Contains, name)                         \
    ->RangeMultiplier(8)                                       \
    ->Ranges({{1 << 15, 1 << 26}})                             \
    ->UseManualTime()                                          \
    ->Unit(benchmark::kMillisecond);

CONTAINS_COLUMN_BENCHMARK_DEFINE(SearchColumn_AllValid, false)
CONTAINS_COLUMN_BENCHMARK_DEFINE(SearchColumn_Nulls, true)
