/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudf/ast/jit_expressions.hpp"

#include "jit/row_ir.hpp"

namespace cudf {
namespace ast {

namespace jit::detail {

cudf::size_type operation::accept(cudf::ast::detail::expression_parser& visitor) const
{
  CUDF_FAIL("predicate is an internal expression and should not be visited by expression_parser",
            std::invalid_argument);
}

std::reference_wrapper<expression const> operation::accept(
  cudf::ast::detail::expression_transformer& visitor) const
{
  CUDF_FAIL(
    "predicate is an internal expression and should not be visited by "
    "expression_transformer",
    std::invalid_argument);
}

bool operation::may_evaluate_null(table_view const& left,
                                  table_view const& right,
                                  rmm::cuda_stream_view stream) const
{
  CUDF_FAIL("predicate is an internal expression and should not be evaluated directly",
            std::invalid_argument);
}

std::unique_ptr<cudf::detail::row_ir::node> operation::accept(
  cudf::detail::row_ir::ast_converter& converter) const
{
  return converter.add_ir_node(*this);
}

}  // namespace jit::detail

expression const& jit::nullify_if(ast::tree& tree, expression const& a, expression const& condition)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::NULLIFY_IF, {a, condition}));
}

expression const& jit::coalesce(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::COALESCE, {a, b}));
}

expression const& jit::predicate(ast::tree& tree, expression const& condition)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::PREDICATE, {condition}));
}

expression const& jit::ansi_add(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_ADD, {a, b}));
}

expression const& jit::ansi_sub(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_SUB, {a, b}));
}

expression const& jit::ansi_mul(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_MUL, {a, b}));
}

expression const& jit::ansi_div(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_DIV, {a, b}));
}

expression const& jit::ansi_mod(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_MOD, {a, b}));
}

expression const& jit::ansi_abs(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_ABS, {a}));
}

expression const& jit::ansi_neg(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_NEG, {a}));
}

expression const& jit::ansi_precision_check(ast::tree& tree,
                                            expression const& a,
                                            expression const& precision)
{
  return tree.push(
    detail::operation(cudf::detail::row_ir::opcode::ANSI_PRECISION_CHECK, {a, precision}));
}

expression const& jit::ansi_try_add(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_TRY_ADD, {a, b}));
}

expression const& jit::ansi_try_sub(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_TRY_SUB, {a, b}));
}

expression const& jit::ansi_try_mul(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_TRY_MUL, {a, b}));
}

expression const& jit::ansi_try_div(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_TRY_DIV, {a, b}));
}

expression const& jit::ansi_try_mod(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_TRY_MOD, {a, b}));
}

expression const& jit::ansi_try_abs(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_TRY_ABS, {a}));
}

expression const& jit::ansi_try_neg(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::ANSI_TRY_NEG, {a}));
}

expression const& jit::ansi_try_precision_check(ast::tree& tree,
                                                expression const& a,
                                                expression const& precision)
{
  return tree.push(
    detail::operation(cudf::detail::row_ir::opcode::ANSI_TRY_PRECISION_CHECK, {a, precision}));
}

expression const& jit::bit_shift_left(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::BIT_SHIFT_LEFT, {a, b}));
}

expression const& jit::bit_shift_right(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::BIT_SHIFT_RIGHT, {a, b}));
}

expression const& jit::cast_to_b8(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_B8, {a}));
}

expression const& jit::cast_to_i8(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_I8, {a}));
}

expression const& jit::cast_to_i16(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_I16, {a}));
}

expression const& jit::cast_to_i32(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_I32, {a}));
}

expression const& jit::cast_to_i64(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_I64, {a}));
}

expression const& jit::cast_to_u8(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_U8, {a}));
}

expression const& jit::cast_to_u16(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_U16, {a}));
}

expression const& jit::cast_to_u32(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_U32, {a}));
}

expression const& jit::cast_to_u64(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_U64, {a}));
}

expression const& jit::cast_to_f32(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_F32, {a}));
}

expression const& jit::cast_to_f64(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_F64, {a}));
}

expression const& jit::cast_to_dec32(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_DEC32, {a}));
}

expression const& jit::cast_to_dec64(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_DEC64, {a}));
}

expression const& jit::cast_to_dec128(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::CAST_TO_DEC128, {a}));
}

expression const& jit::rescale(ast::tree& tree, expression const& a, int32_t target_scale)
{
  return tree.push(detail::operation(cudf::detail::row_ir::opcode::RESCALE, {a}, target_scale));
}

}  // namespace ast
}  // namespace cudf
