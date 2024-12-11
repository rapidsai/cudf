/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <nvbench/nvbench.cuh>
#include <nvbench/types.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <vector>

enum class TreeType {
  IMBALANCED_LEFT  // All operator expressions have a left child operator expression and a right
                   // child column reference
};

template <typename key_type, TreeType tree_type, bool reuse_columns, bool Nullable>
static void BM_ast_transform(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const tree_levels = static_cast<cudf::size_type>(state.get_int64("tree_levels"));

  // Create table data
  auto const n_cols = reuse_columns ? 1 : tree_levels + 1;
  auto const source_table =
    create_sequence_table(cycle_dtypes({cudf::type_to_id<key_type>()}, n_cols),
                          row_count{num_rows},
                          Nullable ? std::optional<double>{0.5} : std::nullopt);
  auto table = source_table->view();

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
  auto expressions = std::list<cudf::ast::operation>();

  // Construct tree that chains additions like (((a + b) + c) + d)
  auto const op = cudf::ast::ast_operator::ADD;
  if (reuse_columns) {
    expressions.push_back(cudf::ast::operation(op, column_refs.at(0), column_refs.at(0)));
    for (cudf::size_type i = 0; i < tree_levels - 1; i++) {
      expressions.push_back(cudf::ast::operation(op, expressions.back(), column_refs.at(0)));
    }
  } else {
    expressions.push_back(cudf::ast::operation(op, column_refs.at(0), column_refs.at(1)));
    std::transform(std::next(column_refs.cbegin(), 2),
                   column_refs.cend(),
                   std::back_inserter(expressions),
                   [&](auto const& column_ref) {
                     return cudf::ast::operation(op, expressions.back(), column_ref);
                   });
  }

  auto const& expression_tree_root = expressions.back();

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(static_cast<size_t>(num_rows) * (tree_levels + 1));
  state.add_global_memory_writes<key_type>(num_rows);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::compute_column(table, expression_tree_root); });
}

template <cudf::ast::ast_operator cmp_op, cudf::ast::ast_operator reduce_op>
static void BM_string_compare_ast_transform(nvbench::state& state)
{
  auto const string_width = static_cast<cudf::size_type>(state.get_int64("string_width"));
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const tree_levels  = static_cast<cudf::size_type>(state.get_int64("tree_levels"));
  auto const hit_rate     = static_cast<cudf::size_type>(state.get_int64("hit_rate"));

  CUDF_EXPECTS(tree_levels > 0, "benchmarks require 1 or more comparisons");

  // Create table data
  auto const num_cols = tree_levels * 2;
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::for_each(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_cols), [&](size_t) {
      columns.emplace_back(create_string_column(num_rows, string_width, hit_rate));
    });

  cudf::table table{std::move(columns)};
  cudf::table_view const table_view = table.view();

  int64_t const chars_size = std::accumulate(
    table_view.begin(),
    table_view.end(),
    static_cast<int64_t>(0),
    [](int64_t size, auto& column) -> int64_t {
      return size + cudf::strings_column_view{column}.chars_size(cudf::get_default_stream());
    });

  // Create column references
  auto column_refs = std::vector<cudf::ast::column_reference>();
  std::transform(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(num_cols),
                 std::back_inserter(column_refs),
                 [](auto const& column_id) { return cudf::ast::column_reference(column_id); });

  // Create expression trees
  std::list<cudf::ast::operation> expressions;

  // Construct AST tree (a == b && c == d && e == f && ...)

  expressions.emplace_back(cudf::ast::operation(cmp_op, column_refs[0], column_refs[1]));

  std::for_each(thrust::make_counting_iterator(1),
                thrust::make_counting_iterator(tree_levels),
                [&](size_t idx) {
                  auto const& lhs = expressions.back();
                  auto const& rhs = expressions.emplace_back(
                    cudf::ast::operation(cmp_op, column_refs[idx * 2], column_refs[idx * 2 + 1]));
                  expressions.emplace_back(cudf::ast::operation(reduce_op, lhs, rhs));
                });

  auto const& expression_tree_root = expressions.back();

  // Use the number of bytes read from global memory
  state.add_element_count(chars_size, "chars_size");
  state.add_global_memory_reads<nvbench::uint8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::compute_column(table, expression_tree_root); });
}

#define AST_TRANSFORM_BENCHMARK_DEFINE(name, key_type, tree_type, reuse_columns, nullable) \
  static void name(::nvbench::state& st)                                                   \
  {                                                                                        \
    ::BM_ast_transform<key_type, tree_type, reuse_columns, nullable>(st);                  \
  }                                                                                        \
  NVBENCH_BENCH(name)                                                                      \
    .set_name(#name)                                                                       \
    .add_int64_axis("tree_levels", {1, 5, 10})                                             \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})

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

#define AST_STRING_COMPARE_TRANSFORM_BENCHMARK_DEFINE(name, cmp_op, reduce_op) \
  static void name(::nvbench::state& st)                                       \
  {                                                                            \
    ::BM_string_compare_ast_transform<cmp_op, reduce_op>(st);                  \
  }                                                                            \
  NVBENCH_BENCH(name)                                                          \
    .set_name(#name)                                                           \
    .add_int64_axis("string_width", {32, 64, 128, 256})                        \
    .add_int64_axis("num_rows", {32768, 262144, 2097152})                      \
    .add_int64_axis("tree_levels", {1, 2, 3, 4})                               \
    .add_int64_axis("hit_rate", {50, 100})

AST_STRING_COMPARE_TRANSFORM_BENCHMARK_DEFINE(ast_string_equal_logical_and,
                                              cudf::ast::ast_operator::EQUAL,
                                              cudf::ast::ast_operator::LOGICAL_AND);
