/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

/**
 * Mathematical unary operations.
 */
public enum UnaryOp {
  SIN(0),
  COS(1),
  TAN(2),
  ARCSIN(3),
  ARCCOS(4),
  ARCTAN(5),
  SINH(6),
  COSH(7),
  TANH(8),
  ARCSINH(9),
  ARCCOSH(10),
  ARCTANH(11),
  EXP(12),
  LOG(13),
  SQRT(14),
  CBRT(15),
  CEIL(16),
  FLOOR(17),
  ABS(18),
  RINT(19),
  BIT_INVERT(20),
  NOT(21);

  private static final UnaryOp[] OPS = UnaryOp.values();
  final int nativeId;

  UnaryOp(int nativeId) {
    this.nativeId = nativeId;
  }

  static UnaryOp fromNative(int nativeId) {
    for (UnaryOp type : OPS) {
      if (type.nativeId == nativeId) {
        return type;
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a UnaryOp");
  }
}
