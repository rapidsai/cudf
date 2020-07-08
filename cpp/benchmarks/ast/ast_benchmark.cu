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

#include <thrust/iterator/counting_iterator.h>

#include <cudf/ast/ast.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>

#include <numeric>
#include <vector>

enum class TreeType {
  DENSE,  // All operator expressions have two child operator expressions (up to the last level)
  LINEAR  // All operator expressions have a left child operator expression (up to the last level)
};

template <typename key_type, TreeType tree_type>
class AST : public cudf::benchmark {
};

template <typename key_type, TreeType tree_type>
static void BM_ast_transform(benchmark::State& state)
{
  const cudf::size_type table_size{(cudf::size_type)state.range(0)};
  const cudf::size_type tree_levels = (cudf::size_type)state.range(1);

  // Create table data
  auto n_cols          = (tree_type == TreeType::DENSE) ? 2 << tree_levels : tree_levels + 1;
  auto column_wrappers = std::vector<cudf::test::fixed_width_column_wrapper<key_type>>();
  auto columns         = std::vector<cudf::column_view>(n_cols);

  auto data_iterator = thrust::make_counting_iterator(0);
  std::generate_n(std::back_inserter(column_wrappers), n_cols, [=]() {
    return cudf::test::fixed_width_column_wrapper<key_type>(data_iterator,
                                                            data_iterator + table_size);
  });
  std::transform(
    column_wrappers.begin(), column_wrappers.end(), columns.begin(), [](auto const& col) {
      return static_cast<cudf::column_view>(col);
    });

  cudf::table_view table{columns};

  auto col_ref_a_0 = cudf::ast::column_reference(0);
  auto col_ref_a_1 = cudf::ast::column_reference(1);
  auto expression_tree =
    cudf::ast::binary_expression(cudf::ast::ast_operator::ADD, col_ref_a_0, col_ref_a_1);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    cudf::ast::compute_column(table, expression_tree);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) * n_cols *
                          sizeof(key_type));
}

#define AST_TRANSFORM_BENCHMARK_DEFINE(name, key_type, tree_type) \
  BENCHMARK_TEMPLATE_DEFINE_F(AST, name, key_type, tree_type)     \
  (::benchmark::State & st) { BM_ast_transform<key_type, tree_type>(st); }

AST_TRANSFORM_BENCHMARK_DEFINE(ast_int32_dense, int32_t, TreeType::DENSE);
AST_TRANSFORM_BENCHMARK_DEFINE(ast_int64_dense, int64_t, TreeType::DENSE);
AST_TRANSFORM_BENCHMARK_DEFINE(ast_float_dense, float, TreeType::DENSE);
AST_TRANSFORM_BENCHMARK_DEFINE(ast_double_dense, double, TreeType::DENSE);
AST_TRANSFORM_BENCHMARK_DEFINE(ast_int32_linear, int32_t, TreeType::LINEAR);
AST_TRANSFORM_BENCHMARK_DEFINE(ast_int64_linear, int64_t, TreeType::LINEAR);
AST_TRANSFORM_BENCHMARK_DEFINE(ast_float_linear, float, TreeType::LINEAR);
AST_TRANSFORM_BENCHMARK_DEFINE(ast_double_linear, double, TreeType::LINEAR);

BENCHMARK_REGISTER_F(AST, ast_int32_dense)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->UseManualTime();
