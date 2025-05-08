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

#include "hybrid_scan_helpers.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

#include <future>
#include <numeric>
#include <optional>

namespace cudf::io::parquet::experimental::detail {

using parquet::detail::chunk_page_info;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::decode_error;
using parquet::detail::PageInfo;

dictionary_literals_collector::dictionary_literals_collector() = default;

dictionary_literals_collector::dictionary_literals_collector(ast::expression const& expr,
                                                             cudf::size_type num_input_columns)
{
  _num_input_columns = num_input_columns;
  _literals.resize(num_input_columns);
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> dictionary_literals_collector::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;
  auto const operands = expr.get_operands();
  auto const op       = expr.get_operator();

  if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
    // First operand should be column reference, second should be literal.
    CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                 "Only binary operations are supported on column reference");
    auto const literal_ptr = dynamic_cast<ast::literal const*>(&operands[1].get());
    CUDF_EXPECTS(literal_ptr != nullptr,
                 "Second operand of binary operation with column reference must be a literal");
    v->accept(*this);

    // Push to the corresponding column's literals and operators list iff EQUAL or NOT_EQUAL
    // operator is seen
    if (op == ast_operator::EQUAL or op == ast::ast_operator::NOT_EQUAL) {
      auto const col_idx = v->get_column_index();
      _literals[col_idx].emplace_back(const_cast<ast::literal*>(literal_ptr));
    }
  } else {
    // Just visit the operands and ignore any output
    std::ignore = visit_operands(operands);
  }

  return expr;
}

}  // namespace cudf::io::parquet::experimental::detail
