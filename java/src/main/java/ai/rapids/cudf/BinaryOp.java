/*
 * Copyright (c) 2019-2020,2022, NVIDIA CORPORATION.
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
package ai.rapids.cudf;

import java.util.EnumSet;

/**
 * Mathematical binary operations.
 */
public enum BinaryOp {
  ADD(0),
  SUB(1),
  MUL(2),
  DIV(3), // divide using common type of lhs and rhs
  TRUE_DIV(4), // divide after promoting to FLOAT64 point
  FLOOR_DIV(5), // divide after promoting to FLOAT64 and flooring the result
  MOD(6),
  PMOD(7), // pmod
  PYMOD(8), // mod operator % follow by python's sign rules for negatives
  POW(9),
  LOG_BASE(10), // logarithm to the base
  ATAN2(11), // atan2
  SHIFT_LEFT(12), // bitwise shift left (<<)
  SHIFT_RIGHT(13), // bitwise shift right (>>)
  SHIFT_RIGHT_UNSIGNED(14), // bitwise shift right (>>>)
  BITWISE_AND(15),
  BITWISE_OR(16),
  BITWISE_XOR(17),
  LOGICAL_AND(18),
  LOGICAL_OR(19),
  EQUAL(20),
  NOT_EQUAL(21),
  LESS(22),
  GREATER(23),
  LESS_EQUAL(24), // <=
  GREATER_EQUAL(25), // >=
  NULL_EQUALS(26), // like EQUAL but NULL == NULL is TRUE and NULL == not NULL is FALSE
  NULL_MAX(27), // MAX but NULL < not NULL
  NULL_MIN(28), // MIN but NULL > not NULL
  //NOT IMPLEMENTED YET GENERIC_BINARY(29);
  NULL_LOGICAL_AND(30),
  NULL_LOGICAL_OR(31);


  static final EnumSet<BinaryOp> COMPARISON = EnumSet.of(
      EQUAL, NOT_EQUAL, LESS, GREATER, LESS_EQUAL, GREATER_EQUAL);
  static final EnumSet<BinaryOp> INEQUALITY_COMPARISON = EnumSet.of(
      LESS, GREATER, LESS_EQUAL, GREATER_EQUAL);

  private static final BinaryOp[] OPS = BinaryOp.values();
  final int nativeId;

  BinaryOp(int nativeId) {
    this.nativeId = nativeId;
  }

  static BinaryOp fromNative(int nativeId) {
    for (BinaryOp type : OPS) {
      if (type.nativeId == nativeId) {
        return type;
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a BinaryOp");
  }
}
