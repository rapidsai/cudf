/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <benchmarks/join/join_common.hpp>

#include <cudf/join/mixed_join.hpp>

auto const num_keys = 2;

// Helper function to create complex AST expressions with multiple operators
void create_complex_ast_expression(cudf::ast::tree& tree, cudf::size_type num_operators)
{
  CUDF_EXPECTS(num_operators > 0, "Number of operators must be greater than 0");

  // For mixed joins, the conditional tables only have 1 column each (column 0)
  // So we'll create multiple comparisons of the same column to stress the AST evaluation

  // Create column references - we only have column 0 available in conditional tables
  tree.push(cudf::ast::column_reference(0));  // index 0: left col 0
  tree.push(
    cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT));  // index 1: right col 0

  // Create first comparison: left_col_0 == right_col_0
  tree.push(cudf::ast::operation(cudf::ast::ast_operator::EQUAL, tree.at(0), tree.at(1)));

  if (num_operators == 1) { return; }

  // For multiple operators, create additional comparisons of the same columns
  // This will create expressions like: (col0_L == col0_R) && (col0_L == col0_R) && ...
  // While this may seem redundant, it stresses the AST evaluation with multiple operators
  for (cudf::size_type i = 1; i < num_operators; i++) {
    // Create another comparison of the same columns
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::EQUAL, tree.at(0), tree.at(1)));

    // Combine with previous result using LOGICAL_AND
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND,
                                   tree.at(tree.size() - 2),  // previous result
                                   tree.back()));             // current comparison
  }
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_mixed_inner_join(nvbench::state& state,
                              nvbench::type_list<nvbench::enum_type<Nullable>,
                                                 nvbench::enum_type<NullEquality>,
                                                 nvbench::enum_type<DataType>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    return cudf::mixed_inner_join(left_equality_input,
                                  right_equality_input,
                                  left_conditional_input,
                                  right_conditional_input,
                                  binary_pred,
                                  compare_nulls);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_mixed_left_join(nvbench::state& state,
                             nvbench::type_list<nvbench::enum_type<Nullable>,
                                                nvbench::enum_type<NullEquality>,
                                                nvbench::enum_type<DataType>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    return cudf::mixed_left_join(left_equality_input,
                                 right_equality_input,
                                 left_conditional_input,
                                 right_conditional_input,
                                 binary_pred,
                                 compare_nulls);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_mixed_full_join(nvbench::state& state,
                             nvbench::type_list<nvbench::enum_type<Nullable>,
                                                nvbench::enum_type<NullEquality>,
                                                nvbench::enum_type<DataType>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    return cudf::mixed_full_join(left_equality_input,
                                 right_equality_input,
                                 left_conditional_input,
                                 right_conditional_input,
                                 binary_pred,
                                 compare_nulls);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_mixed_left_semi_join(nvbench::state& state,
                                  nvbench::type_list<nvbench::enum_type<Nullable>,
                                                     nvbench::enum_type<NullEquality>,
                                                     nvbench::enum_type<DataType>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    return cudf::mixed_left_semi_join(left_equality_input,
                                      right_equality_input,
                                      left_conditional_input,
                                      right_conditional_input,
                                      binary_pred,
                                      compare_nulls);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_mixed_left_anti_join(nvbench::state& state,
                                  nvbench::type_list<nvbench::enum_type<Nullable>,
                                                     nvbench::enum_type<NullEquality>,
                                                     nvbench::enum_type<DataType>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    return cudf::mixed_left_anti_join(left_equality_input,
                                      right_equality_input,
                                      left_conditional_input,
                                      right_conditional_input,
                                      binary_pred,
                                      compare_nulls);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_mixed_inner_join_complex_ast(nvbench::state& state,
                                          nvbench::type_list<nvbench::enum_type<Nullable>,
                                                             nvbench::enum_type<NullEquality>,
                                                             nvbench::enum_type<DataType>>)
{
  auto const num_ast_operators = static_cast<cudf::size_type>(state.get_int64("ast_operators"));

  auto join = [num_ast_operators](cudf::table_view const& left_equality_input,
                                  cudf::table_view const& right_equality_input,
                                  cudf::table_view const& left_conditional_input,
                                  cudf::table_view const& right_conditional_input,
                                  cudf::ast::operation binary_pred,
                                  cudf::null_equality compare_nulls) {
    // Create complex AST expression with multiple operators
    cudf::ast::tree tree;
    create_complex_ast_expression(tree, num_ast_operators);

    return cudf::mixed_inner_join(left_equality_input,
                                  right_equality_input,
                                  left_conditional_input,
                                  right_conditional_input,
                                  tree.back(),
                                  compare_nulls);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

NVBENCH_BENCH_TYPES(nvbench_mixed_inner_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("mixed_inner_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(nvbench_mixed_inner_join_complex_ast,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("mixed_inner_join_complex_ast")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("ast_operators", {1, 5, 10});

NVBENCH_BENCH_TYPES(nvbench_mixed_left_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("mixed_left_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(nvbench_mixed_full_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("mixed_full_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(nvbench_mixed_left_semi_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("mixed_left_semi_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH_TYPES(nvbench_mixed_left_anti_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("mixed_left_anti_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE);
