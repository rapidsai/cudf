
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudf/ast/expressions.hpp"

#include "ast/jit/expressions.hpp"
#include "jit/row_ir.hpp"

namespace cudf {
namespace ast {
namespace jit {
namespace detail {

cudf::size_type operation::accept(cudf::ast::detail::expression_parser& visitor) const
{
  CUDF_FAIL(
    "JIT operation is an internal expression and should not be visited by expression_parser",
    std::invalid_argument);
}

std::reference_wrapper<expression const> operation::accept(
  cudf::ast::detail::expression_transformer& visitor) const
{
  CUDF_FAIL(
    "JIT operation is an internal expression and should not be visited by "
    "expression_transformer",
    std::invalid_argument);
}

bool operation::may_evaluate_null(table_view const& left,
                                  table_view const& right,
                                  rmm::cuda_stream_view stream) const
{
  CUDF_FAIL("JIT operation is an internal expression and should not be evaluated directly",
            std::invalid_argument);
}

std::unique_ptr<cudf::detail::row_ir::node> operation::accept(
  cudf::detail::row_ir::ast_converter& converter) const
{
  return converter.add_ir_node(*this);
}

}  // namespace detail
}  // namespace jit

using opcode = cudf::detail::row_ir::opcode;

namespace {

[[nodiscard]] opcode as_opcode(cudf::ast::jit::op operator_id)
{
  using enum cudf::ast::jit::op;
  switch (operator_id) {
    case COALESCE: return opcode::COALESCE;
    case PREDICATE: return opcode::PREDICATE;
    case ADD: return opcode::ADD;
    case ADD_OVERFLOW: return opcode::ADD_OVERFLOW;
    case SUB: return opcode::SUB;
    case SUB_OVERFLOW: return opcode::SUB_OVERFLOW;
    case MUL: return opcode::MUL;
    case MUL_OVERFLOW: return opcode::MUL_OVERFLOW;
    case DIV: return opcode::DIV;
    case DIV_OVERFLOW: return opcode::DIV_OVERFLOW;
    case MOD: return opcode::MOD;
    case MOD_OVERFLOW: return opcode::MOD_OVERFLOW;
    case ABS: return opcode::ABS;
    case ABS_OVERFLOW: return opcode::ABS_OVERFLOW;
    case NEG: return opcode::NEG;
    case NEG_OVERFLOW: return opcode::NEG_OVERFLOW;
    case CHECK_PRECISION: return opcode::CHECK_PRECISION;
    case BITWISE_SHIFT_LEFT: return opcode::BITWISE_SHIFT_LEFT;
    case BITWISE_SHIFT_RIGHT: return opcode::BITWISE_SHIFT_RIGHT;
    case CAST_TO_BOOL8: return opcode::CAST_TO_BOOL8;
    case CAST_TO_INT8: return opcode::CAST_TO_INT8;
    case CAST_TO_INT16: return opcode::CAST_TO_INT16;
    case CAST_TO_INT32: return opcode::CAST_TO_INT32;
    case CAST_TO_INT64: return opcode::CAST_TO_INT64;
    case CAST_TO_UINT8: return opcode::CAST_TO_UINT8;
    case CAST_TO_UINT16: return opcode::CAST_TO_UINT16;
    case CAST_TO_UINT32: return opcode::CAST_TO_UINT32;
    case CAST_TO_UINT64: return opcode::CAST_TO_UINT64;
    case CAST_TO_FLOAT32: return opcode::CAST_TO_FLOAT32;
    case CAST_TO_FLOAT64: return opcode::CAST_TO_FLOAT64;
    case CAST_TO_DECIMAL32: return opcode::CAST_TO_DECIMAL32;
    case CAST_TO_DECIMAL64: return opcode::CAST_TO_DECIMAL64;
    case CAST_TO_DECIMAL128: return opcode::CAST_TO_DECIMAL128;
    case RESCALE: return opcode::RESCALE;
    default: CUDF_FAIL("Invalid JIT op", std::invalid_argument);
  }
}

}  // namespace

expression const& jit::operation(ast::tree& tree,
                                 op operator_id,
                                 std::vector<std::reference_wrapper<expression const>> args,
                                 bool nullify_on_error,
                                 std::optional<int32_t> target_scale)
{
  auto ir_opcode = as_opcode(operator_id);

  if (target_scale.has_value()) {
    CUDF_EXPECTS(operator_id == op::RESCALE,
                 "target_scale is only valid for jit::op::RESCALE",
                 std::invalid_argument);
    return tree.push(
      detail::operation(ir_opcode, std::move(args), target_scale.value(), nullify_on_error));
  }

  CUDF_EXPECTS(operator_id != op::RESCALE,
               "target_scale must be set for jit::op::RESCALE",
               std::invalid_argument);
  return tree.push(detail::operation(ir_opcode, std::move(args), nullify_on_error));
}

expression const& jit::operation(
  ast::tree& tree,
  op operator_id,
  std::initializer_list<std::reference_wrapper<expression const>> args,
  bool nullify_on_error,
  std::optional<int32_t> target_scale)
{
  return operation(tree,
                   operator_id,
                   std::vector<std::reference_wrapper<expression const>>{args},
                   nullify_on_error,
                   target_scale);
}

}  // namespace ast
}  // namespace cudf
