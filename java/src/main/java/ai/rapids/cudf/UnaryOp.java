/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  BIT_COUNT(20),
  BIT_INVERT(21),
  NOT(22);

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
