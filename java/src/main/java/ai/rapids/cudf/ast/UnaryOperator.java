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

/**
 * Enumeration of AST operators that can appear in a unary operation.
 * NOTE: This must be kept in sync with `jni_to_unary_operator` in CompiledExpression.cpp!
 */
public enum UnaryOperator {
  IDENTITY(0),          // Identity function
  SIN(1),               // Trigonometric sine
  COS(2),               // Trigonometric cosine
  TAN(3),               // Trigonometric tangent
  ARCSIN(4),            // Trigonometric sine inverse
  ARCCOS(5),            // Trigonometric cosine inverse
  ARCTAN(6),            // Trigonometric tangent inverse
  SINH(7),              // Hyperbolic sine
  COSH(8),              // Hyperbolic cosine
  TANH(9),              // Hyperbolic tangent
  ARCSINH(10),          // Hyperbolic sine inverse
  ARCCOSH(11),          // Hyperbolic cosine inverse
  ARCTANH(12),          // Hyperbolic tangent inverse
  EXP(13),              // Exponential (base e, Euler number)
  LOG(14),              // Natural Logarithm (base e)
  SQRT(15),             // Square-root (x^0.5)
  CBRT(16),             // Cube-root (x^(1.0/3))
  CEIL(17),             // Smallest integer value not less than arg
  FLOOR(18),            // largest integer value not greater than arg
  ABS(19),              // Absolute value
  RINT(20),             // Rounds the floating-point argument arg to an integer value
  BIT_INVERT(21),       // Bitwise Not (~)
  NOT(22),              // Logical Not (!)
  CAST_TO_INT64(23),    // Cast value to int64_t
  CAST_TO_UINT64(24),   // Cast value to uint64_t
  CAST_TO_FLOAT64(25);  // Cast value to double

  private final byte nativeId;

  UnaryOperator(int nativeId) {
    this.nativeId = (byte) nativeId;
    assert this.nativeId == nativeId;
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
