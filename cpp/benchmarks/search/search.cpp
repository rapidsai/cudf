/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

class Search : public cudf::benchmark {};

void BM_column(benchmark::State& state, bool nulls)
{
  auto const column_size{static_cast<cudf::size_type>(state.range(0))};
  auto const values_size = column_size;

  auto init_data  = cudf::make_fixed_width_scalar<float>(static_cast<float>(0));
  auto init_value = cudf::make_fixed_width_scalar<float>(static_cast<float>(values_size));
  auto step       = cudf::make_fixed_width_scalar<float>(static_cast<float>(-1));
  auto column     = cudf::sequence(column_size, *init_data);
  auto values     = cudf::sequence(values_size, *init_value, *step);
  if (nulls) {
    auto [column_null_mask, column_null_count] = create_random_null_mask(column->size(), 0.1, 1);
    column->set_null_mask(std::move(column_null_mask), column_null_count);
    auto [values_null_mask, values_null_count] = create_random_null_mask(values->size(), 0.1, 2);
    values->set_null_mask(std::move(values_null_mask), values_null_count);
  }

  auto data_table = cudf::sort(cudf::table_view({*column}));

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto col = cudf::upper_bound(data_table->view(),
                                 cudf::table_view({*values}),
                                 {cudf::order::ASCENDING},
                                 {cudf::null_order::BEFORE});
  }
}

BENCHMARK_DEFINE_F(Search, Column_AllValid)(::benchmark::State& state) { BM_column(state, false); }
BENCHMARK_DEFINE_F(Search, Column_Nulls)(::benchmark::State& state) { BM_column(state, true); }

BENCHMARK_REGISTER_F(Search, Column_AllValid)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Arg(100000000);

BENCHMARK_REGISTER_F(Search, Column_Nulls)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Arg(100000000);

void BM_table(benchmark::State& state)
{
  using Type = float;

  auto const num_columns{static_cast<cudf::size_type>(state.range(0))};
  auto const column_size{static_cast<cudf::size_type>(state.range(1))};
  auto const values_size = column_size;

  data_profile profile = data_profile_builder().cardinality(0).null_probability(0.1).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 100);
  auto data_table = create_random_table(
    cycle_dtypes({cudf::type_to_id<Type>()}, num_columns), row_count{column_size}, profile);
  auto values_table = create_random_table(
    cycle_dtypes({cudf::type_to_id<Type>()}, num_columns), row_count{values_size}, profile);

  std::vector<cudf::order> orders(num_columns, cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_orders(num_columns, cudf::null_order::BEFORE);
  auto sorted = cudf::sort(*data_table);

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto col = cudf::lower_bound(sorted->view(), *values_table, orders, null_orders);
  }
}

BENCHMARK_DEFINE_F(Search, Table)(::benchmark::State& state) { BM_table(state); }

static void CustomArguments(benchmark::internal::Benchmark* b)
{
  for (int num_cols = 1; num_cols <= 10; num_cols *= 2)
    for (int col_size = 1000; col_size <= 100000000; col_size *= 10)
      b->Args({num_cols, col_size});
}

BENCHMARK_REGISTER_F(Search, Table)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Apply(CustomArguments);

void BM_contains(benchmark::State& state, bool nulls)
{
  auto const column_size{static_cast<cudf::size_type>(state.range(0))};
  auto const values_size = column_size;

  auto init_data  = cudf::make_fixed_width_scalar<float>(static_cast<float>(0));
  auto init_value = cudf::make_fixed_width_scalar<float>(static_cast<float>(values_size));
  auto step       = cudf::make_fixed_width_scalar<float>(static_cast<float>(-1));
  auto column     = cudf::sequence(column_size, *init_data);
  auto values     = cudf::sequence(values_size, *init_value, *step);
  if (nulls) {
    auto [column_null_mask, column_null_count] = create_random_null_mask(column->size(), 0.1, 1);
    column->set_null_mask(std::move(column_null_mask), column_null_count);
    auto [values_null_mask, values_null_count] = create_random_null_mask(values->size(), 0.1, 2);
    values->set_null_mask(std::move(values_null_mask), values_null_count);
  }

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto col = cudf::contains(*column, *values);
  }
}

BENCHMARK_DEFINE_F(Search, ColumnContains_AllValid)(::benchmark::State& state)
{
  BM_contains(state, false);
}
BENCHMARK_DEFINE_F(Search, ColumnContains_Nulls)(::benchmark::State& state)
{
  BM_contains(state, true);
}

BENCHMARK_REGISTER_F(Search, ColumnContains_AllValid)
  ->RangeMultiplier(8)
  ->Ranges({{1 << 10, 1 << 26}})
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Search, ColumnContains_Nulls)
  ->RangeMultiplier(8)
  ->Ranges({{1 << 10, 1 << 26}})
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);
