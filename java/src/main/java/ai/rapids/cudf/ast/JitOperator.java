/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/**
 * The subset of libcudf row-IR operators exposed through the Java AST API. Standard comparison,
 * logical, and mathematical operations remain available through {@link BinaryOperation} and
 * {@link UnaryOperation}.
 * Operators ending in {@code _OVERFLOW} and {@link #CHECK_PRECISION} are fallible and support
 * {@link JitErrorPolicy#NULLIFY}. Other operators require the default
 * {@link JitErrorPolicy#PROPAGATE} policy.
 *
 * NOTE: This must be kept in sync with `jni_to_jit_operator` in CompiledExpression.cpp!
 */
public enum JitOperator {
  /** Return the first non-null input. */
  COALESCE(0, 2, false, false),
  /** Convert a nullable boolean input into an always-valid predicate. */
  PREDICATE(1, 1, false, false),
  ADD(2, 2, false, false),
  SUB(3, 2, false, false),
  MUL(4, 2, false, false),
  /** Divide without reporting arithmetic errors. */
  DIV(5, 2, false, false),
  NEG(6, 1, false, false),
  ABS(7, 1, false, false),
  MOD(8, 2, false, false),
  ADD_OVERFLOW(9, 2, true, false),
  SUB_OVERFLOW(10, 2, true, false),
  MUL_OVERFLOW(11, 2, true, false),
  /** Divide and report division-by-zero and signed-overflow errors. */
  DIV_OVERFLOW(12, 2, true, false),
  NEG_OVERFLOW(13, 1, true, false),
  ABS_OVERFLOW(14, 1, true, false),
  MOD_OVERFLOW(15, 2, true, false),
  /** Verify that decimal input 0 fits the INT32 precision supplied by input 1. */
  CHECK_PRECISION(16, 2, true, false),
  BITWISE_SHIFT_LEFT(17, 2, false, false),
  BITWISE_SHIFT_RIGHT(18, 2, false, false),
  CAST_TO_BOOL8(19, 1, false, false),
  CAST_TO_INT8(20, 1, false, false),
  CAST_TO_INT16(21, 1, false, false),
  CAST_TO_INT32(22, 1, false, false),
  CAST_TO_INT64(23, 1, false, false),
  CAST_TO_UINT8(24, 1, false, false),
  CAST_TO_UINT16(25, 1, false, false),
  CAST_TO_UINT32(26, 1, false, false),
  CAST_TO_UINT64(27, 1, false, false),
  CAST_TO_FLOAT32(28, 1, false, false),
  CAST_TO_FLOAT64(29, 1, false, false),
  CAST_TO_DECIMAL32(30, 1, false, false),
  CAST_TO_DECIMAL64(31, 1, false, false),
  CAST_TO_DECIMAL128(32, 1, false, false),
  /** Rescale a decimal input to the target scale supplied to {@link JitOperation}. */
  RESCALE(33, 1, false, true),
  /** Select input 0 when input 2 is true, otherwise input 1. */
  IF_ELSE(34, 3, false, false);

  private final byte nativeId;
  private final int arity;
  private final boolean fallible;
  private final boolean requiresTargetScale;

  JitOperator(int nativeId, int arity, boolean fallible, boolean requiresTargetScale) {
    this.nativeId = (byte) nativeId;
    this.arity = arity;
    this.fallible = fallible;
    this.requiresTargetScale = requiresTargetScale;
    assert this.nativeId == nativeId;
  }

  int getArity() {
    return arity;
  }

  boolean isFallible() {
    return fallible;
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
