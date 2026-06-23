
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

expression const& jit::coalesce(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(cudf::ast::jit::detail::operation(opcode::COALESCE, {a, b}));
}

expression const& jit::predicate(ast::tree& tree, expression const& condition)
{
  return tree.push(detail::operation(opcode::PREDICATE, {condition}));
}

expression const& jit::add(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(opcode::ADD, {a, b}, false));
}

expression const& jit::add_overflow(ast::tree& tree,
                                    expression const& a,
                                    expression const& b,
                                    bool nullify_on_error)
{
  return tree.push(detail::operation(opcode::ADD_OVERFLOW, {a, b}, nullify_on_error));
}

expression const& jit::sub(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(opcode::SUB, {a, b}, false));
}

expression const& jit::sub_overflow(ast::tree& tree,
                                    expression const& a,
                                    expression const& b,
                                    bool nullify_on_error)
{
  return tree.push(detail::operation(opcode::SUB_OVERFLOW, {a, b}, nullify_on_error));
}

expression const& jit::mul(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(opcode::MUL, {a, b}, false));
}

expression const& jit::mul_overflow(ast::tree& tree,
                                    expression const& a,
                                    expression const& b,
                                    bool nullify_on_error)
{
  return tree.push(detail::operation(opcode::MUL_OVERFLOW, {a, b}, nullify_on_error));
}

expression const& jit::div(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(opcode::DIV, {a, b}, false));
}

expression const& jit::div_overflow(ast::tree& tree,
                                    expression const& a,
                                    expression const& b,
                                    bool nullify_on_error)
{
  return tree.push(detail::operation(opcode::DIV_OVERFLOW, {a, b}, nullify_on_error));
}

expression const& jit::mod(ast::tree& tree, expression const& a, expression const& b)
{
  return tree.push(detail::operation(opcode::MOD, {a, b}, false));
}

expression const& jit::mod_overflow(ast::tree& tree,
                                    expression const& a,
                                    expression const& b,
                                    bool nullify_on_error)
{
  return tree.push(detail::operation(opcode::MOD_OVERFLOW, {a, b}, nullify_on_error));
}

expression const& jit::abs(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::ABS, {a}, false));
}

expression const& jit::abs_overflow(ast::tree& tree, expression const& a, bool nullify_on_error)
{
  return tree.push(detail::operation(opcode::ABS_OVERFLOW, {a}, nullify_on_error));
}

expression const& jit::neg(ast::tree& tree, expression const& a)
{
  return tree.push(detail::operation(opcode::NEG, {a}, false));
}

expression const& jit::neg_overflow(ast::tree& tree, expression const& a, bool nullify_on_error)
{
  return tree.push(detail::operation(opcode::NEG_OVERFLOW, {a}, nullify_on_error));
}

expression const& jit::check_precision(ast::tree& tree,
                                       expression const& a,
                                       expression const& precision,
                                       bool nullify_on_error)
{
  return tree.push(detail::operation(opcode::CHECK_PRECISION, {a, precision}, nullify_on_error));
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
