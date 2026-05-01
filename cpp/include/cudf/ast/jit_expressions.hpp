/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/operators/opcodes.hpp>

namespace CUDF_EXPORT cudf {
namespace ast {

/**
 * @addtogroup expressions
 * @{
 * @file
 */

namespace jit {
namespace detail {

struct operation : public ast::expression {
  /**
* @brief Construct a new operation object.
* @param op The opcode for this operation
* @param args The arguments for this operation
  */
  operation(cudf::detail::row_ir::opcode op,
            std::vector<std::reference_wrapper<expression const>> args)
    : op_{op}, args_{std::move(args)}
  {
  }

  /**
  * @brief Construct a new operation object with a target scale (for rescale and precision check
  * operations).
  * @param op The opcode for this operation
  * @param args The arguments for this operation
  * @param target_scale The target scale for this operation (only applicable for rescale and precision check operations)
   */
  operation(cudf::detail::row_ir::opcode op,
            std::vector<std::reference_wrapper<expression const>> args,
            int32_t target_scale)
    : op_{op}, args_{std::move(args)}, target_scale_{target_scale}
  {
  }

  operation(operation const&)            = default; //< Copy constructor
  operation(operation&&)                 = default; //< Move constructor
  operation& operator=(operation const&) = default; //< Copy assignment
  operation& operator=(operation&&)      = default; //< Move assignment
  ~operation() override                  = default; //< Destructor

  /**
   * @brief Get the opcode.
   *
   * @return The opcode
   */
  [[nodiscard]] cudf::detail::row_ir::opcode get_opcode() const { return op_; }

  /**
   * @brief Get the operands.
   *
   * @return Vector of operands
   */
  [[nodiscard]] std::span<std::reference_wrapper<expression const> const> get_arguments() const
  {
    return args_;
  }

  /**
   * @brief Get the target scale for rescale and precision check operations.
   *
   * @return The target scale if applicable, std::nullopt otherwise
   */
  [[nodiscard]] std::optional<int32_t> get_target_scale() const { return target_scale_; }

  /**
   * @copydoc expression::accept
   */
  cudf::size_type accept(cudf::ast::detail::expression_parser& visitor) const override;

  /**
   * @copydoc expression::accept
   */
  std::reference_wrapper<expression const> accept(
    cudf::ast::detail::expression_transformer& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override;

  /**
   * @copydoc expression::accept
   */
  [[nodiscard]] std::unique_ptr<cudf::detail::row_ir::node> accept(
    cudf::detail::row_ir::ast_converter& visitor) const override;

 private:
  cudf::detail::row_ir::opcode op_;
  std::vector<std::reference_wrapper<expression const>> args_;
  std::optional<int32_t> target_scale_ = std::nullopt;
};

}  // namespace detail

expression const& nullify_if(ast::tree& tree, expression const& condition);

expression const& coalesce(ast::tree& tree, expression const& a, expression const& b);

expression const& predicate(ast::tree& tree, expression const& condition);

expression const& ansi_add(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_sub(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_mul(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_div(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_mod(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_abs(ast::tree& tree, expression const& a);

expression const& ansi_neg(ast::tree& tree, expression const& a);

expression const& ansi_precision_check(ast::tree& tree, expression const& a, int32_t precision);

expression const& ansi_try_add(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_try_sub(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_try_mul(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_try_div(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_try_mod(ast::tree& tree, expression const& a, expression const& b);

expression const& ansi_try_abs(ast::tree& tree, expression const& a);

expression const& ansi_try_neg(ast::tree& tree, expression const& a);

expression const& ansi_try_precision_check(ast::tree& tree, expression const& a, int32_t precision);

expression const& bit_shift_left(ast::tree& tree, expression const& a, expression const& b);

expression const& bit_shift_right(ast::tree& tree, expression const& a, expression const& b);

expression const& cast_to_b8(ast::tree& tree, expression const& a);

expression const& cast_to_i8(ast::tree& tree, expression const& a);

expression const& cast_to_i16(ast::tree& tree, expression const& a);

expression const& cast_to_i32(ast::tree& tree, expression const& a);

expression const& cast_to_i64(ast::tree& tree, expression const& a);

expression const& cast_to_u8(ast::tree& tree, expression const& a);

expression const& cast_to_u16(ast::tree& tree, expression const& a);

expression const& cast_to_u32(ast::tree& tree, expression const& a);

expression const& cast_to_u64(ast::tree& tree, expression const& a);

expression const& cast_to_f32(ast::tree& tree, expression const& a);

expression const& cast_to_f64(ast::tree& tree, expression const& a);

expression const& cast_to_dec32(ast::tree& tree, expression const& a);

expression const& cast_to_dec64(ast::tree& tree, expression const& a);

expression const& cast_to_dec128(ast::tree& tree, expression const& a);

expression const& rescale(ast::tree& tree, expression const& a, int32_t new_scale);

}  // namespace jit

}  // namespace ast
}  // namespace CUDF_EXPORT cudf
