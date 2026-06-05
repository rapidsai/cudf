/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/utilities/export.hpp>

#include <cstdint>

namespace CUDF_EXPORT cudf {
namespace ast {
namespace jit {

/**
 * @brief Enumeration for specifying the mode of arithmetic operations (DEFAULT, ANSI, or ANSI_TRY).
 */
enum class compliance_mode : uint8_t {
  //< DEFAULT mode performs arithmetic operations using DEFAULT behavior (e.g., allowing overflow,
  // silent truncation, and undefined behavior).
  DEFAULT,
  //< ANSI mode performs arithmetic operations using ANSI behavior (e.g., throwing an error on
  // overflow).
  ANSI,
  //< ANSI_TRY mode performs arithmetic operations using ANSI behavior but returns NULL on error
  // instead of throwing.
  ANSI_TRY
};

/**
 * @brief Creates an expression that evaluates to the first non-null value among its arguments.
 * @param tree The expression tree to which this expression will be added
 * @param a The first expression to coalesce
 * @param b The second expression to coalesce
 * @return An expression representing the coalesced value
 */
expression const& coalesce(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that evaluates to `true` if the condition is true and not null,
 * and `false` otherwise. This is used to implement predicates in the JIT.
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
 * @param mode The mode to use for the addition operation (DEFAULT, ANSI, or ANSI_TRY)
 * @return An expression representing the result of the addition
 */
expression const& add(ast::tree& tree,
                      expression const& a,
                      expression const& b,
                      compliance_mode mode = compliance_mode::DEFAULT);

/**
 * @brief Creates an expression that performs ANSI-compliant subtraction of `a` and `b`, which
 * throws an error on overflow.
 * @param tree The expression tree to which this expression will be added
 * @param a The minuend
 * @param b The subtrahend
 * @param mode The mode to use for the subtraction operation (DEFAULT, ANSI, or ANSI_TRY)
 * @return An expression representing the result of the subtraction
 */
expression const& sub(ast::tree& tree,
                      expression const& a,
                      expression const& b,
                      compliance_mode mode = compliance_mode::DEFAULT);

/**
 * @brief Creates an expression that performs ANSI-compliant multiplication of `a` and `b`, which
 * throws an error on overflow.
 * @param tree The expression tree to which this expression will be added
 * @param a The first factor
 * @param b The second factor
 * @param mode The mode to use for the multiplication operation (DEFAULT, ANSI, or ANSI_TRY)
 * @return An expression representing the result of the multiplication
 */
expression const& mul(ast::tree& tree,
                      expression const& a,
                      expression const& b,
                      compliance_mode mode = compliance_mode::DEFAULT);

/**
 * @brief Creates an expression that performs ANSI-compliant division of `a` by `b`, which throws
 * an error on division by zero.
 * @param tree The expression tree to which this expression will be added
 * @param a The dividend
 * @param b The divisor
 * @param mode The mode to use for the division operation (DEFAULT, ANSI, or ANSI_TRY)
 * @return An expression representing the result of the division
 */
expression const& div(ast::tree& tree,
                      expression const& a,
                      expression const& b,
                      compliance_mode mode = compliance_mode::DEFAULT);

/**
 * @brief Creates an expression that performs ANSI-compliant modulus of `a` by `b`, which throws
 * an error on division by zero.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to be divided
 * @param b The divisor
 * @param mode The mode to use for the modulus operation (DEFAULT, ANSI, or ANSI_TRY)
 * @return An expression representing the result of the modulus operation
 */
expression const& mod(ast::tree& tree,
                      expression const& a,
                      expression const& b,
                      compliance_mode mode = compliance_mode::DEFAULT);

/**
 * @brief Creates an expression that performs ANSI-compliant absolute value of `a`, which throws
 * an error on overflow.
 * @param tree The expression tree to which this expression will be added
 * @param a The value for which to compute the absolute value
 * @param mode The mode to use for the absolute value operation (DEFAULT, ANSI, or ANSI_TRY)
 * error on overflow
 * @return An expression representing the absolute value
 */
expression const& abs(ast::tree& tree,
                      expression const& a,
                      compliance_mode mode = compliance_mode::DEFAULT);

/**
 * @brief Creates an expression that performs ANSI-compliant negation of `a`, which throws an
 * error on overflow.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to negate
 * @param mode The mode to use for the negation operation (DEFAULT, ANSI, or ANSI_TRY)
 * @return An expression representing the negated value
 */
expression const& neg(ast::tree& tree,
                      expression const& a,
                      compliance_mode mode = compliance_mode::DEFAULT);

/**
 * @brief Creates an expression that performs an ANSI-compliant precision check on `a` with the
 * given precision, which throws an error if the value of `a` exceeds the specified precision.
 * @param tree The expression tree to which this expression will be added
 * @param a The value for which to perform the precision check
 * @param precision The precision to check against
 * @param mode The mode to use for the precision check operation (DEFAULT, ANSI, or ANSI_TRY)
 * @return An expression representing the result of the precision check
 */
expression const& precision_check(ast::tree& tree,
                                  expression const& a,
                                  expression const& precision,
                                  compliance_mode mode = compliance_mode::DEFAULT);

/**
 * @brief Creates an expression that performs a bitwise left shift of `a` by `b`.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to shift
 * @param b The number of bits by which to shift
 * @return An expression representing the result of the bitwise left shift
 */
expression const& bitwise_shift_left(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs a bitwise right shift of `a` by `b`.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to shift
 * @param b The number of bits by which to shift
 * @return An expression representing the result of the bitwise right shift
 */
expression const& bitwise_shift_right(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that casts `a` to a boolean type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_bool8(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 8-bit signed integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_int8(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 16-bit signed integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_int16(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 32-bit signed integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_int32(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 64-bit signed integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_int64(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to an 8-bit unsigned integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_uint8(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 16-bit unsigned integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_uint16(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 32-bit unsigned integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_uint32(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 64-bit unsigned integer type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_uint64(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 32-bit floating point type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_float32(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 64-bit floating point type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_float64(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 32-bit decimal type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_decimal32(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 64-bit decimal type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_decimal64(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that casts `a` to a 128-bit decimal type.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to cast
 * @return An expression representing the result of the cast
 */
expression const& cast_to_decimal128(ast::tree& tree, expression const& a);

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
