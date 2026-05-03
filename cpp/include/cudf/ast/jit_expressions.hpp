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
   * @param target_scale The target scale for this operation (only applicable for rescale and
   * precision check operations)
   */
  operation(cudf::detail::row_ir::opcode op,
            std::vector<std::reference_wrapper<expression const>> args,
            int32_t target_scale)
    : op_{op}, args_{std::move(args)}, target_scale_{target_scale}
  {
  }

  operation(operation const&)            = default;  //< Copy constructor
  operation(operation&&)                 = default;  //< Move constructor
  operation& operator=(operation const&) = default;  //< Copy assignment
  operation& operator=(operation&&)      = default;  //< Move assignment
  ~operation() override                  = default;  //< Destructor

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

/**
 * @brief Creates an expression that evaluates to `NULL` if the condition is true, and the value of
 * `a` otherwise.
 * @param tree The expression tree to which this expression will be added
 * @param a The expression to nullify if the condition is true
 * @param condition The condition under which to nullify the value
 * @return An expression representing the nullified value
 */
expression const& nullify_if(ast::tree& tree, expression const& a,  expression const& condition);

/**
 * @brief Creates an expression that evaluates to the first non-null value among its arguments.
 * @param tree The expression tree to which this expression will be added
 * @param a The first expression to coalesce
 * @param b The second expression to coalesce
 * @return An expression representing the coalesced value
 */
expression const& coalesce(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that evaluates to `true` if the condition is true and not null, and
 * `false` otherwise. This is used to implement predicates in the JIT.
 * @param tree The expression tree to which this expression will be added
 * @param condition The condition to evaluate as a predicate
 * @return An expression representing the result of the predicate
 */
expression const& predicate(ast::tree& tree, expression const& condition);

/**
 * @brief Creates an expression that performs ANSI-compliant addition of `a` and `b`, which throws
 * an error on overflow.
 * @param tree The expression tree to which this expression will be added
 * @param a The first addend
 * @param b The second addend
 * @return An expression representing the result of the addition
 */
expression const& ansi_add(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant subtraction of `a` and `b`, which
 * throws an error on overflow.
 * @param tree The expression tree to which this expression will be added
 * @param a The minuend
 * @param b The subtrahend
 * @return An expression representing the result of the subtraction
 */
expression const& ansi_sub(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant multiplication of `a` and `b`, which
 * throws an error on overflow.
 * @param tree The expression tree to which this expression will be added
 * @param a The first factor
 * @param b The second factor
 * @return An expression representing the result of the multiplication
 */
expression const& ansi_mul(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant division of `a` by `b`, which throws an
 * error on division by zero.
 * @param tree The expression tree to which this expression will be added
 * @param a The dividend
 * @param b The divisor
 * @return An expression representing the result of the division
 */
expression const& ansi_div(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant modulus of `a` by `b`, which throws an
 * error on division by zero.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to be divided
 * @param b The divisor
 * @return An expression representing the result of the modulus operation
 */
expression const& ansi_mod(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant absolute value of `a`, which throws an
 * error on overflow.
 * @param tree The expression tree to which this expression will be added
 * @param a The value for which to compute the absolute value
 * @return An expression representing the absolute value
 */
expression const& ansi_abs(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that performs ANSI-compliant negation of `a`, which throws an error
 * on overflow.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to negate
 * @return An expression representing the negated value
 */
expression const& ansi_neg(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that performs an ANSI-compliant precision check on `a` with the
 * given precision, which throws an error if the value of `a` exceeds the specified precision.
 * @param tree The expression tree to which this expression will be added
 * @param a The value for which to perform the precision check
 * @param precision The precision to check against
 * @return An expression representing the result of the precision check
 */
expression const& ansi_precision_check(ast::tree& tree,
                                       expression const& a,
                                       expression const& precision);

/**
 * @brief Creates an expression that performs ANSI-compliant addition of `a` and `b`, which returns
 * `NULL` on overflow instead of throwing an error.
 * @param tree The expression tree to which this expression will be added
 * @param a The first addend
 * @param b The second addend
 * @return An expression representing the result of the addition, or `NULL` if overflow occurs
 */
expression const& ansi_try_add(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant subtraction of `a` and `b`, which
 * returns `NULL` on overflow instead of throwing an error.
 * @param tree The expression tree to which this expression will be added
 * @param a The minuend
 * @param b The subtrahend
 * @return An expression representing the result of the subtraction, or `NULL` if overflow occurs
 */
expression const& ansi_try_sub(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant multiplication of `a` and `b`, which
 * returns `NULL` on overflow instead of throwing an error.
 * @param tree The expression tree to which this expression will be added
 * @param a The first factor
 * @param b The second factor
 * @return An expression representing the result of the multiplication, or `NULL` if overflow occurs
 */
expression const& ansi_try_mul(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant division of `a` by `b`, which returns
 * `NULL` on division by zero instead of throwing an error.
 * @param tree The expression tree to which this expression will be added
 * @param a The dividend
 * @param b The divisor
 * @return An expression representing the result of the division, or `NULL` if division by zero
 * occurs
 */
expression const& ansi_try_div(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant modulus of `a` by `b`, which returns
 * `NULL` on division by zero instead of throwing an error.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to be divided
 * @param b The divisor
 * @return An expression representing the result of the modulus operation, or `NULL` if division by
 * zero occurs
 */
expression const& ansi_try_mod(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs ANSI-compliant absolute value of `a`, which returns
 * `NULL` on overflow instead of throwing an error.
 * @param tree The expression tree to which this expression will be added
 * @param a The value for which to compute the absolute value
 * @return An expression representing the absolute value, or `NULL` if overflow occurs
 */
expression const& ansi_try_abs(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that performs ANSI-compliant negation of `a`, which returns `NULL`
 * on overflow instead of throwing an error.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to negate
 * @return An expression representing the negated value, or `NULL` if overflow occurs
 */
expression const& ansi_try_neg(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that performs an ANSI-compliant precision check on `a` with the
 * given precision, which returns `NULL` if the value of `a` exceeds the specified precision instead
 * of throwing an error.
 * @param tree The expression tree to which this expression will be added
 * @param a The value for which to perform the precision check
 * @param precision The precision to check against
 * @return An expression representing the result of the precision check, or `NULL` if the value
 * exceeds the specified precision
 */
expression const& ansi_try_precision_check(ast::tree& tree,
                                           expression const& a,
                                           expression const& precision);

/**
 * @brief Creates an expression that performs a bitwise left shift of `a` by `b`.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to shift
 * @param b The number of bits by which to shift
 * @return An expression representing the result of the bitwise left shift
 */
expression const& bit_shift_left(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs a bitwise right shift of `a` by `b`.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to shift
 * @param b The number of bits by which to shift
 * @return An expression representing the result of the bitwise right shift
 */
expression const& bit_shift_right(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that casts `a` to a boolean type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_b8(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 8-bit signed integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_i8(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 16-bit signed integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_i16(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 32-bit signed integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_i32(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 64-bit signed integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_i64(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to an 8-bit unsigned integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_u8(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 16-bit unsigned integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_u16(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 32-bit unsigned integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_u32(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 64-bit unsigned integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_u64(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 32-bit floating point type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_f32(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 64-bit floating point type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_f64(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 32-bit decimal type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_dec32(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 64-bit decimal type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_dec64(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 128-bit decimal type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_dec128(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that rescales a decimal expression `a` to a new scale `new_scale`.
 * @param tree The expression tree to which this expression will be added
 * @param a The decimal expression to rescale
 * @param new_scale The new scale to which to rescale the decimal expression
 * @return An expression representing the rescaled decimal value
 */
expression const& rescale(ast::tree& tree, expression const& a, int32_t new_scale);

}  // namespace jit

}  // namespace ast
}  // namespace CUDF_EXPORT cudf
