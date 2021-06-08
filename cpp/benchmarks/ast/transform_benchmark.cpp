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

#include <cudf/ast/transform.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <benchmark/benchmark.h>
#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <list>
#include <numeric>
#include <random>
#include <vector>

enum class TreeType {
  IMBALANCED_LEFT  // All operator expressions have a left child operator expression and a right
                   // child column reference
};

template <typename key_type, TreeType tree_type, bool reuse_columns, bool Nullable>
class AST : public cudf::benchmark {
};

template <typename key_type, TreeType tree_type, bool reuse_columns, bool Nullable>
static void BM_ast_transform(benchmark::State& state)
{
  const cudf::size_type table_size{(cudf::size_type)state.range(0)};
  const cudf::size_type tree_levels = (cudf::size_type)state.range(1);

  // Create table data
  auto n_cols          = reuse_columns ? 1 : tree_levels + 1;
  auto column_wrappers = std::vector<cudf::test::fixed_width_column_wrapper<key_type>>(n_cols);
  auto columns         = std::vector<cudf::column_view>(n_cols);

  auto data_iterator = thrust::make_counting_iterator(0);

  if constexpr (Nullable) {
    auto validities = std::vector<bool>(table_size);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::generate(
      validities.begin(), validities.end(), [&]() { return gen() > (0.5 * gen.max()); });
    std::generate_n(column_wrappers.begin(), n_cols, [=]() {
      return cudf::test::fixed_width_column_wrapper<key_type>(
        data_iterator, data_iterator + table_size, validities.begin());
    });
  } else {
    std::generate_n(column_wrappers.begin(), n_cols, [=]() {
      return cudf::test::fixed_width_column_wrapper<key_type>(data_iterator,
                                                              data_iterator + table_size);
    });
  }
  std::transform(
    column_wrappers.begin(), column_wrappers.end(), columns.begin(), [](auto const& col) {
      return static_cast<cudf::column_view>(col);
    });

  cudf::table_view table{columns};

  // Create column references
  auto column_refs = std::vector<cudf::ast::column_reference>();
  std::transform(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(n_cols),
                 std::back_inserter(column_refs),
                 [](auto const& column_id) {
                   return cudf::ast::column_reference(reuse_columns ? 0 : column_id);
                 });

  // Create expression trees

  // Note that a std::list is required here because of its guarantees against reference invalidation
  // when items are added or removed. References to items in a std::vector are not safe if the
  // vector must re-allocate.
  auto expressions = std::list<cudf::ast::expression>();

  // Construct tree that chains additions like (((a + b) + c) + d)
  auto const op = cudf::ast::ast_operator::ADD;
  if (reuse_columns) {
    expressions.push_back(cudf::ast::expression(op, column_refs.at(0), column_refs.at(0)));
    for (cudf::size_type i = 0; i < tree_levels - 1; i++) {
      expressions.push_back(cudf::ast::expression(op, expressions.back(), column_refs.at(0)));
    }
  } else {
    expressions.push_back(cudf::ast::expression(op, column_refs.at(0), column_refs.at(1)));
    std::transform(std::next(column_refs.cbegin(), 2),
                   column_refs.cend(),
                   std::back_inserter(expressions),
                   [&](auto const& column_ref) {
                     return cudf::ast::expression(op, expressions.back(), column_ref);
                   });
  }

  auto const& expression_tree_root = expressions.back();

  // Execute benchmark
  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::ast::compute_column(table, expression_tree_root);
  }

  // Use the number of bytes read from global memory
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          (tree_levels + 1) * sizeof(key_type));
}

#define AST_TRANSFORM_BENCHMARK_DEFINE(name, key_type, tree_type, reuse_columns, nullable) \
  BENCHMARK_TEMPLATE_DEFINE_F(AST, name, key_type, tree_type, reuse_columns, nullable)     \
  (::benchmark::State & st) { BM_ast_transform<key_type, tree_type, reuse_columns, nullable>(st); }

AST_TRANSFORM_BENCHMARK_DEFINE(
  ast_int32_imbalanced_unique, int32_t, TreeType::IMBALANCED_LEFT, false, false);
AST_TRANSFORM_BENCHMARK_DEFINE(
  ast_int32_imbalanced_reuse, int32_t, TreeType::IMBALANCED_LEFT, true, false);
AST_TRANSFORM_BENCHMARK_DEFINE(
  ast_double_imbalanced_unique, double, TreeType::IMBALANCED_LEFT, false, false);

AST_TRANSFORM_BENCHMARK_DEFINE(
  ast_int32_imbalanced_unique_nulls, int32_t, TreeType::IMBALANCED_LEFT, false, true);
AST_TRANSFORM_BENCHMARK_DEFINE(
  ast_int32_imbalanced_reuse_nulls, int32_t, TreeType::IMBALANCED_LEFT, true, true);
AST_TRANSFORM_BENCHMARK_DEFINE(
  ast_double_imbalanced_unique_nulls, double, TreeType::IMBALANCED_LEFT, false, true);

static void CustomRanges(benchmark::internal::Benchmark* b)
{
  auto row_counts       = std::vector<cudf::size_type>{100'000, 1'000'000, 10'000'000, 100'000'000};
  auto operation_counts = std::vector<cudf::size_type>{1, 5, 10};
  for (auto const& row_count : row_counts) {
    for (auto const& operation_count : operation_counts) { b->Args({row_count, operation_count}); }
  }
}

BENCHMARK_REGISTER_F(AST, ast_int32_imbalanced_unique)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

BENCHMARK_REGISTER_F(AST, ast_int32_imbalanced_reuse)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

BENCHMARK_REGISTER_F(AST, ast_double_imbalanced_unique)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

BENCHMARK_REGISTER_F(AST, ast_int32_imbalanced_unique_nulls)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

BENCHMARK_REGISTER_F(AST, ast_int32_imbalanced_reuse_nulls)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();

BENCHMARK_REGISTER_F(AST, ast_double_imbalanced_unique_nulls)
  ->Apply(CustomRanges)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime();
