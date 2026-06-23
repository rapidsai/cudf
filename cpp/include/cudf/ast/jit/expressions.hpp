/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief Creates an expression that performs addition of `a` and `b`.
 * @param tree The expression tree to which this expression will be added
 * @param a The first addend
 * @param b The second addend
 * @return An expression representing the result of the addition
 */
expression const& add(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs addition of `a` and `b` with overflow
 * handling.
 * @param tree The expression tree to which this expression will be added
 * @param a The first addend
 * @param b The second addend
 * @param nullify_on_error If true, returns NULL on overflow; if false, throws an error on overflow
 * @return An expression representing the result of the addition
 */
expression const& add_overflow(ast::tree& tree,
                               expression const& a,
                               expression const& b,
                               bool nullify_on_error = false);

/**
 * @brief Creates an expression that performs subtraction of `a` and `b`.
 * @param tree The expression tree to which this expression will be added
 * @param a The minuend
 * @param b The subtrahend
 * @return An expression representing the result of the subtraction
 */
expression const& sub(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs subtraction of `a` and `b` with
 * overflow handling.
 * @param tree The expression tree to which this expression will be added
 * @param a The minuend
 * @param b The subtrahend
 * @param nullify_on_error If true, returns NULL on overflow; if false, throws an error on overflow
 * @return An expression representing the result of the subtraction
 */
expression const& sub_overflow(ast::tree& tree,
                               expression const& a,
                               expression const& b,
                               bool nullify_on_error = false);

/**
 * @brief Creates an expression that performs multiplication of `a` and `b`.
 * @param tree The expression tree to which this expression will be added
 * @param a The first factor
 * @param b The second factor
 * @return An expression representing the result of the multiplication
 */
expression const& mul(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs multiplication of `a` and `b` with
 * overflow handling.
 * @param tree The expression tree to which this expression will be added
 * @param a The first factor
 * @param b The second factor
 * @param nullify_on_error If true, returns NULL on overflow; if false, throws an error on overflow
 * @return An expression representing the result of the multiplication
 */
expression const& mul_overflow(ast::tree& tree,
                               expression const& a,
                               expression const& b,
                               bool nullify_on_error = false);

/**
 * @brief Creates an expression that performs division of `a` by `b`.
 * @param tree The expression tree to which this expression will be added
 * @param a The dividend
 * @param b The divisor
 * @return An expression representing the result of the division
 */
expression const& div(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs division of `a` by `b` with overflow
 * handling.
 * @param tree The expression tree to which this expression will be added
 * @param a The dividend
 * @param b The divisor
 * @param nullify_on_error If true, returns NULL on division by zero; if false, throws an error on
 * division by zero
 * @return An expression representing the result of the division
 */
expression const& div_overflow(ast::tree& tree,
                               expression const& a,
                               expression const& b,
                               bool nullify_on_error = false);

/**
 * @brief Creates an expression that performs modulus of `a` by `b`.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to be divided
 * @param b The divisor
 * @return An expression representing the result of the modulus operation
 */
expression const& mod(ast::tree& tree, expression const& a, expression const& b);

/**
 * @brief Creates an expression that performs modulus of `a` by `b` with overflow
 * handling.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to be divided
 * @param b The divisor
 * @param nullify_on_error If true, returns NULL on division by zero; if false, throws an error on
 * division by zero
 * @return An expression representing the result of the modulus operation
 */
expression const& mod_overflow(ast::tree& tree,
                               expression const& a,
                               expression const& b,
                               bool nullify_on_error = false);

/**
 * @brief Creates an expression that performs absolute value of `a`.
 * @param tree The expression tree to which this expression will be added
 * @param a The value for which to compute the absolute value
 * @return An expression representing the absolute value
 */
expression const& abs(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that performs absolute value of `a` with overflow
 * handling.
 * @param tree The expression tree to which this expression will be added
 * @param a The value for which to compute the absolute value
 * @param nullify_on_error If true, returns NULL on overflow; if false, throws an error on overflow
 * @return An expression representing the absolute value
 */
expression const& abs_overflow(ast::tree& tree, expression const& a, bool nullify_on_error = false);

/**
 * @brief Creates an expression that performs negation of `a`.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to negate
 * @return An expression representing the negated value
 */
expression const& neg(ast::tree& tree, expression const& a);

/**
 * @brief Creates an expression that performs negation of `a` with overflow
 * handling.
 * @param tree The expression tree to which this expression will be added
 * @param a The value to negate
 * @param nullify_on_error If true, returns NULL on overflow; if false, throws an error on overflow
 * @return An expression representing the negated value
 */
expression const& neg_overflow(ast::tree& tree, expression const& a, bool nullify_on_error = false);

/**
 * @brief Creates an expression that performs a precision check on `a` with the given precision.
 * @param tree The expression tree to which this expression will be added
 * @param a The value for which to perform the precision check
 * @param precision The precision to check against
 * @param nullify_on_error If true, returns NULL on precision overflow; if false, throws an error
 * @return An expression representing the result of the precision check
 */
expression const& check_precision(ast::tree& tree,
                                  expression const& a,
                                  expression const& precision,
                                  bool nullify_on_error = false);

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
