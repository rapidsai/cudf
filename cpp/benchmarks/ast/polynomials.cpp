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
static void BM_ast_polynomials(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const order    = static_cast<cudf::size_type>(state.get_int64("order"));

  CUDF_EXPECTS(order > 0, "Polynomial order must be greater than 0");

  data_profile profile;
  profile.set_distribution_params(cudf::type_to_id<key_type>(),
                                  distribution_id::NORMAL,
                                  static_cast<key_type>(0),
                                  static_cast<key_type>(1));
  auto table = create_random_table({cudf::type_to_id<key_type>()}, row_count{num_rows}, profile);
  auto column_view = table->get_column(0);

  std::vector<cudf::numeric_scalar<key_type>> constants;
  {
    std::random_device random_device;
    std::mt19937 generator;
    std::uniform_real_distribution<key_type> distribution{0, 1};

    std::transform(thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(order + 1),
                   std::back_inserter(constants),
                   [&](int) { return distribution(generator); });
  }

  cudf::ast::tree tree{};

  auto& column_ref = tree.push(cudf::ast::column_reference{0});

  // computes polynomials: (((ax + b)x + c)x + d)x + e... = ax**4 + bx**3 + cx**2 + dx + e....
  tree.push(cudf::ast::literal{constants[0]});

  for (cudf::size_type i = 0; i < order; i++) {
    auto& product =
      tree.push(cudf::ast::operation{cudf::ast::ast_operator::MUL, tree.back(), column_ref});
    auto& constant = tree.push(cudf::ast::literal{constants[i + 1]});
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::ADD, product, constant});
  }

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(num_rows);
  state.add_global_memory_writes<key_type>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::scoped_range range{"benchmark_iteration"};
    cudf::compute_column(*table, tree.back(), launch.get_stream().get_stream());
  });
}

#define AST_POLYNOMIAL_BENCHMARK_DEFINE(name, key_type)                          \
  static void name(::nvbench::state& st) { ::BM_ast_polynomials<key_type>(st); } \
  NVBENCH_BENCH(name)                                                            \
    .set_name(#name)                                                             \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})   \
    .add_int64_axis("order", {1, 2, 4, 8, 16, 32})

AST_POLYNOMIAL_BENCHMARK_DEFINE(ast_polynomials_float32, float);

AST_POLYNOMIAL_BENCHMARK_DEFINE(ast_polynomials_float64, double);
