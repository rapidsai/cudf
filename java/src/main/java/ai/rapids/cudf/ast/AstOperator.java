/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;
import java.util.EnumSet;

/**
 * Enumeration of AST operations that can appear in an expression
 * NOTE: This must be kept in sync with the `jni_to_ast_operator` code in CompiledExpression.cpp!
 */
public enum AstOperator {
  // Binary operators
  ADD(0),            // operator +
  SUB(1),            // operator -
  MUL(2),            // operator *
  DIV(3),            // operator / using common type of lhs and rhs
  TRUE_DIV(4),       // operator / after promoting type to floating point
  FLOOR_DIV(5),      // operator / after promoting to 64 bit floating point and then flooring the result
  MOD(6),            // operator %
  PYMOD(7),          // operator % but following python's sign rules for negatives
  POW(8),            // lhs ^ rhs
  EQUAL(9),          // operator ==
  NOT_EQUAL(10),     // operator !=
  LESS(11),          // operator <
  GREATER(12),       // operator >
  LESS_EQUAL(13),    // operator <=
  GREATER_EQUAL(14), // operator >=
  BITWISE_AND(15),   // operator &
  BITWISE_OR(16),    // operator |
  BITWISE_XOR(17),   // operator ^
  LOGICAL_AND(18),   // operator &&
  LOGICAL_OR(19),    // operator ||
  // Unary operators
  IDENTITY(20),      // Identity function
  SIN(21),           // Trigonometric sine
  COS(22),           // Trigonometric cosine
  TAN(23),           // Trigonometric tangent
  ARCSIN(24),        // Trigonometric sine inverse
  ARCCOS(25),        // Trigonometric cosine inverse
  ARCTAN(26),        // Trigonometric tangent inverse
  SINH(27),          // Hyperbolic sine
  COSH(28),          // Hyperbolic cosine
  TANH(29),          // Hyperbolic tangent
  ARCSINH(30),       // Hyperbolic sine inverse
  ARCCOSH(31),       // Hyperbolic cosine inverse
  ARCTANH(32),       // Hyperbolic tangent inverse
  EXP(33),           // Exponential (base e, Euler number)
  LOG(34),           // Natural Logarithm (base e)
  SQRT(35),          // Square-root (x^0.5)
  CBRT(36),          // Cube-root (x^(1.0/3))
  CEIL(37),          // Smallest integer value not less than arg
  FLOOR(38),         // largest integer value not greater than arg
  ABS(39),           // Absolute value
  RINT(40),          // Rounds the floating-point argument arg to an integer value
  BIT_INVERT(41),    // Bitwise Not (~)
  NOT(42);           // Logical Not (!)

  private static final EnumSet<AstOperator> unaryOps = EnumSet.of(
      IDENTITY,
      SIN,
      COS,
      TAN,
      ARCSIN,
      ARCCOS,
      ARCTAN,
      SINH,
      COSH,
      TANH,
      ARCSINH,
      ARCCOSH,
      ARCTANH,
      EXP,
      LOG,
      SQRT,
      CBRT,
      CEIL,
      FLOOR,
      ABS,
      RINT,
      BIT_INVERT,
      NOT);

  private static final EnumSet<AstOperator> binaryOps = EnumSet.of(
      ADD,
      SUB,
      MUL,
      DIV,
      TRUE_DIV,
      FLOOR_DIV,
      MOD,
      PYMOD,
      POW,
      EQUAL,
      NOT_EQUAL,
      LESS,
      GREATER,
      LESS_EQUAL,
      GREATER_EQUAL,
      BITWISE_AND,
      BITWISE_OR,
      BITWISE_XOR,
      LOGICAL_AND,
      LOGICAL_OR);

  private final byte nativeId;

  AstOperator(int nativeId) {
    this.nativeId = (byte) nativeId;
    assert this.nativeId == nativeId;
  }

  boolean isUnaryOperator() {
    return unaryOps.contains(this);
  }

  boolean isBinaryOperator() {
    return binaryOps.contains(this);
  }

  /** Get the size in bytes to serialize this operator */
  int getSerializedSize() {
    return Byte.BYTES;
  }

  /** Serialize this operator to the specified buffer */
  void serialize(ByteBuffer bb) {
    bb.put(nativeId);
  }
}
