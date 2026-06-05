
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudf/ast/jit/expressions.hpp"

#include "ast/jit/expressions.hpp"
#include "jit/row_ir.hpp"

namespace cudf {
namespace ast {
namespace jit {
namespace detail {

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

}  // namespace detail
}  // namespace jit

using opcode = cudf::detail::row_ir::opcode;

expression const& jit::coalesce(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(cudf::ast::jit::detail::operation(opcode::COALESCE, {a, b}));
}

expression const& jit::predicate(ast::tree& tree, expression const& condition)
{
  return tree.push(detail::operation(opcode::PREDICATE, {condition}));
}

std::tuple<opcode, bool> resolve_op(opcode default_op, opcode ansi_op, jit::compliance_mode mode)
{
  switch (mode) {
    case jit::compliance_mode::DEFAULT: return {default_op, false};
    case jit::compliance_mode::ANSI: return {ansi_op, false};
    case jit::compliance_mode::ANSI_TRY: return {ansi_op, true};
    default: CUDF_FAIL("Invalid compliance mode", std::invalid_argument);
  }
}

expression const& jit::add(ast::tree& tree,
                           expression const& a,
                           expression const& b,
                           compliance_mode mode)
{
  auto [op, nullify_on_error] = resolve_op(opcode::ADD, opcode::ANSI_ADD, mode);

  return tree.push(detail::operation(op, {a, b}, nullify_on_error));
}

expression const& jit::sub(ast::tree& tree,
                           expression const& a,
                           expression const& b,
                           compliance_mode mode)
{
  auto [op, nullify_on_error] = resolve_op(opcode::SUB, opcode::ANSI_SUB, mode);

  return tree.push(detail::operation(op, {a, b}, nullify_on_error));
}

expression const& jit::mul(ast::tree& tree,
                           expression const& a,
                           expression const& b,
                           compliance_mode mode)
{
  auto [op, nullify_on_error] = resolve_op(opcode::MUL, opcode::ANSI_MUL, mode);

  return tree.push(detail::operation(op, {a, b}, nullify_on_error));
}

expression const& jit::div(ast::tree& tree,
                           expression const& a,
                           expression const& b,
                           compliance_mode mode)
{
  auto [op, nullify_on_error] = resolve_op(opcode::DIV, opcode::ANSI_DIV, mode);

  return tree.push(detail::operation(op, {a, b}, nullify_on_error));
}

expression const& jit::mod(ast::tree& tree,
                           expression const& a,
                           expression const& b,
                           compliance_mode mode)
{
  auto [op, nullify_on_error] = resolve_op(opcode::MOD, opcode::ANSI_MOD, mode);

  return tree.push(detail::operation(op, {a, b}, nullify_on_error));
}

expression const& jit::abs(ast::tree& tree, expression const& a, compliance_mode mode)
{
  auto [op, nullify_on_error] = resolve_op(opcode::ABS, opcode::ANSI_ABS, mode);

  return tree.push(detail::operation(op, {a}, nullify_on_error));
}

expression const& jit::neg(ast::tree& tree, expression const& a, compliance_mode mode)
{
  auto [op, nullify_on_error] = resolve_op(opcode::NEG, opcode::ANSI_NEG, mode);

  return tree.push(detail::operation(op, {a}, nullify_on_error));
}

expression const& jit::precision_check(ast::tree& tree,
                                       expression const& a,
                                       expression const& precision,
                                       compliance_mode mode)
{
  opcode op             = opcode::IDENTITY;
  bool nullify_on_error = false;

  switch (mode) {
    case compliance_mode::DEFAULT: {
      op               = opcode::IDENTITY;
      nullify_on_error = false;
    } break;
    case compliance_mode::ANSI: {
      op               = opcode::ANSI_PRECISION_CHECK;
      nullify_on_error = false;
    } break;
    case compliance_mode::ANSI_TRY: {
      op               = opcode::ANSI_PRECISION_CHECK;
      nullify_on_error = true;
    } break;
    default: CUDF_FAIL("Invalid compliance mode for precision check", std::invalid_argument);
  }

  return tree.push(detail::operation(op, {a, precision}, nullify_on_error));
}

expression const& jit::bitwise_shift_left(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(opcode::BITWISE_SHIFT_LEFT, {a, b}));
}

expression const& jit::bitwise_shift_right(ast::tree& tree,
                                           expression const& a,
                                           expression const& b)
{
  return tree.push(detail::operation(opcode::BITWISE_SHIFT_RIGHT, {a, b}));
}

expression const& jit::cast_to_bool8(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_BOOL8, {a}));
}

expression const& jit::cast_to_int8(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_INT8, {a}));
}

expression const& jit::cast_to_int16(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_INT16, {a}));
}

expression const& jit::cast_to_int32(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_INT32, {a}));
}

expression const& jit::cast_to_int64(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_INT64, {a}));
}

expression const& jit::cast_to_uint8(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_UINT8, {a}));
}

expression const& jit::cast_to_uint16(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_UINT16, {a}));
}

expression const& jit::cast_to_uint32(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_UINT32, {a}));
}

expression const& jit::cast_to_uint64(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_UINT64, {a}));
}

expression const& jit::cast_to_float32(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_FLOAT32, {a}));
}

expression const& jit::cast_to_float64(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_FLOAT64, {a}));
}

expression const& jit::cast_to_decimal32(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_DECIMAL32, {a}));
}

expression const& jit::cast_to_decimal64(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_DECIMAL64, {a}));
}

expression const& jit::cast_to_decimal128(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::CAST_TO_DECIMAL128, {a}));
}

expression const& jit::rescale(ast::tree& tree, expression const& a, int32_t target_scale)
{
  return tree.push(detail::operation(opcode::RESCALE, {a}, target_scale));
}

}  // namespace ast
}  // namespace cudf
