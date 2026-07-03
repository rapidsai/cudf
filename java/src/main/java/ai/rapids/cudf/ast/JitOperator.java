/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/**
 * Enumeration of AST JIT operators backed by libcudf row IR opcodes.
 * NOTE: This must be kept in sync with `compile_jit_expression` in CompiledExpression.cpp!
 */
public enum JitOperator {
  ADD(0, 2, true, false),
  SUB(1, 2, true, false),
  MUL(2, 2, true, false),
  DIV(3, 2, true, false),
  MOD(4, 2, true, false),
  ABS(5, 1, true, false),
  NEG(6, 1, true, false),
  PRECISION_CHECK(7, 2, true, false),
  BITWISE_SHIFT_LEFT(8, 2, false, false),
  BITWISE_SHIFT_RIGHT(9, 2, false, false),
  COALESCE(10, 2, false, false),
  PREDICATE(11, 1, false, false),
  CAST_TO_BOOL8(12, 1, false, false),
  CAST_TO_INT8(13, 1, false, false),
  CAST_TO_INT16(14, 1, false, false),
  CAST_TO_INT32(15, 1, false, false),
  CAST_TO_INT64(16, 1, false, false),
  CAST_TO_UINT8(17, 1, false, false),
  CAST_TO_UINT16(18, 1, false, false),
  CAST_TO_UINT32(19, 1, false, false),
  CAST_TO_UINT64(20, 1, false, false),
  CAST_TO_FLOAT32(21, 1, false, false),
  CAST_TO_FLOAT64(22, 1, false, false),
  CAST_TO_DECIMAL32(23, 1, false, false),
  CAST_TO_DECIMAL64(24, 1, false, false),
  CAST_TO_DECIMAL128(25, 1, false, false),
  RESCALE(26, 1, false, true),
  IF_ELSE(27, 3, false, false);

  private final byte nativeId;
  private final int arity;
  private final boolean supportsComplianceMode;
  private final boolean requiresTargetScale;

  JitOperator(int nativeId, int arity, boolean supportsComplianceMode, boolean requiresTargetScale) {
    this.nativeId = (byte) nativeId;
    this.arity = arity;
    this.supportsComplianceMode = supportsComplianceMode;
    this.requiresTargetScale = requiresTargetScale;
    assert this.nativeId == nativeId;
  }

  int getArity() {
    return arity;
  }

  boolean supportsComplianceMode() {
    return supportsComplianceMode;
  }

  boolean requiresTargetScale() {
    return requiresTargetScale;
  }

  int getSerializedSize() {
    return Byte.BYTES;
  }

  void serialize(ByteBuffer bb) {
    bb.put(nativeId);
  }
}
