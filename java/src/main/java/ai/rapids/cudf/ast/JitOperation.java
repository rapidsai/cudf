/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;
import java.util.Objects;

/** A JIT operation consisting of a row IR opcode and operands. */
public final class JitOperation extends AstExpression {
  private final JitOperator op;
  private final JitComplianceMode complianceMode;
  private final AstExpression[] inputs;
  private final Integer targetScale;

  public JitOperation(JitOperator op, AstExpression... inputs) {
    this(op, JitComplianceMode.DEFAULT, null, inputs);
  }

  public JitOperation(JitOperator op, JitComplianceMode complianceMode, AstExpression... inputs) {
    this(op, complianceMode, null, inputs);
  }

  public JitOperation(JitOperator op, int targetScale, AstExpression... inputs) {
    this(op, JitComplianceMode.DEFAULT, Integer.valueOf(targetScale), inputs);
  }

  private JitOperation(
      JitOperator op,
      JitComplianceMode complianceMode,
      Integer targetScale,
      AstExpression... inputs) {
    this.op = Objects.requireNonNull(op, "op is null");
    this.complianceMode = Objects.requireNonNull(complianceMode, "complianceMode is null");
    this.inputs = Objects.requireNonNull(inputs, "inputs is null").clone();
    this.targetScale = targetScale;
    if (this.inputs.length != op.getArity()) {
      throw new IllegalArgumentException(
          op + " requires " + op.getArity() + " inputs, found " + this.inputs.length);
    }
    if (!op.supportsComplianceMode() && complianceMode != JitComplianceMode.DEFAULT) {
      throw new IllegalArgumentException(op + " does not support compliance mode " + complianceMode);
    }
    if (op == JitOperator.PRECISION_CHECK && complianceMode == JitComplianceMode.DEFAULT) {
      throw new IllegalArgumentException("PRECISION_CHECK requires ANSI or ANSI_TRY compliance mode");
    }
    if (op.requiresTargetScale() != (targetScale != null)) {
      throw new IllegalArgumentException(op + " target scale usage is invalid");
    }
    for (AstExpression input : this.inputs) {
      Objects.requireNonNull(input, "input is null");
    }
  }

  @Override
  int getSerializedSize() {
    int size = ExpressionType.JIT_EXPRESSION.getSerializedSize() +
        op.getSerializedSize() +
        complianceMode.getSerializedSize() +
        Byte.BYTES +
        Byte.BYTES;
    if (targetScale != null) {
      size += Integer.BYTES;
    }
    for (AstExpression input : inputs) {
      size += input.getSerializedSize();
    }
    return size;
  }

  @Override
  void serialize(ByteBuffer bb) {
    ExpressionType.JIT_EXPRESSION.serialize(bb);
    op.serialize(bb);
    complianceMode.serialize(bb);
    bb.put((byte) inputs.length);
    bb.put((byte) (targetScale == null ? 0 : 1));
    if (targetScale != null) {
      bb.putInt(targetScale);
    }
    for (AstExpression input : inputs) {
      input.serialize(bb);
    }
  }

  @Override
  public String toString() {
    StringBuilder ret = new StringBuilder(op.toString());
    if (complianceMode != JitComplianceMode.DEFAULT) {
      ret.append("[").append(complianceMode).append("]");
    }
    ret.append("(");
    for (int i = 0; i < inputs.length; i++) {
      if (i > 0) {
        ret.append(", ");
      }
      ret.append(inputs[i]);
    }
    if (targetScale != null) {
      if (inputs.length > 0) {
        ret.append(", ");
      }
      ret.append("scale=").append(targetScale);
    }
    return ret.append(")").toString();
  }
}
