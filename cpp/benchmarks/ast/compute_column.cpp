/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <nvbench/nvbench.cuh>
#include <nvbench/types.cuh>

#include <random>

template <typename key_type>
static void BM_ast_compute_column(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  data_profile profile;
  profile.set_distribution_params(cudf::type_to_id<key_type>(),
                                  distribution_id::NORMAL,
                                  static_cast<key_type>(0),
                                  static_cast<key_type>(1));
  auto table = create_random_table({cudf::type_to_id<key_type>()}, row_count{num_rows}, profile);
  auto column_view = table->get_column(0);

  std::vector<cudf::numeric_scalar<key_type>> constants{
    cudf::numeric_scalar<key_type>{2},
    cudf::numeric_scalar<key_type>{3}
  };

  cudf::ast::tree tree{};

  auto& column_ref = tree.push(cudf::ast::column_reference{0});

  // computes polynomial: ax + b
  auto& product = tree.push(cudf::ast::operation{cudf::ast::ast_operator::MUL, column_ref, tree.push(cudf::ast::literal{constants[0]})});
  tree.push(cudf::ast::operation{cudf::ast::ast_operator::ADD, product, tree.push(cudf::ast::literal{constants[1]})});

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(num_rows);
  state.add_global_memory_writes<key_type>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::scoped_range range{"benchmark_iteration"};
    cudf::compute_column(*table, tree.back(), launch.get_stream().get_stream());
  });
}

#define AST_COMPUTE_COLUMN_BENCHMARK_DEFINE(name, key_type)                          \
  static void name(::nvbench::state& st) { ::BM_ast_compute_column<key_type>(st); } \
  NVBENCH_BENCH(name)                                                            \
    .set_name(#name)                                                             \
    .add_int64_axis("num_rows", {600'000'000})

AST_COMPUTE_COLUMN_BENCHMARK_DEFINE(ast_compute_column_float32, float);

AST_COMPUTE_COLUMN_BENCHMARK_DEFINE(ast_compute_column_float64, double);
