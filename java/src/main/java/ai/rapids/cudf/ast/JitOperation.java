/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * A libcudf row-IR operation. Expressions containing a JIT operation must be evaluated with
 * {@link CompiledExpression#computeColumnJit}.
 */
public final class JitOperation extends AstExpression {
  private final JitOperator op;
  private final JitErrorPolicy errorPolicy;
  private final AstExpression[] inputs;
  private final Integer targetScale;

  /**
   * Construct an operation that propagates evaluation errors.
   *
   * @param op operator to apply
   * @param inputs operator inputs
   * @throws NullPointerException if {@code op}, {@code inputs}, or an input is null
   * @throws IllegalArgumentException if the operator arity or target-scale usage is invalid
   */
  public JitOperation(JitOperator op, AstExpression... inputs) {
    this(op, JitErrorPolicy.PROPAGATE, null, inputs);
  }

  /**
   * Construct an operation with an explicit error policy.
   *
   * @param op operator to apply
   * @param errorPolicy error handling policy
   * @param inputs operator inputs
   * @throws NullPointerException if any argument or input is null
   * @throws IllegalArgumentException if the arity, policy, or target-scale usage is invalid
   */
  public JitOperation(JitOperator op, JitErrorPolicy errorPolicy, AstExpression... inputs) {
    this(op, errorPolicy, null, inputs);
  }

  /**
   * Construct a target-scale operation that propagates evaluation errors.
   * The target scale is valid only for {@link JitOperator#RESCALE}.
   *
   * @param op operator to apply
   * @param targetScale target fixed-point scale
   * @param inputs operator inputs
   * @throws NullPointerException if {@code op}, {@code inputs}, or an input is null
   * @throws IllegalArgumentException if the operator is not {@link JitOperator#RESCALE} or its
   *         arity is invalid
   */
  public JitOperation(JitOperator op, int targetScale, AstExpression... inputs) {
    this(op, JitErrorPolicy.PROPAGATE, Integer.valueOf(targetScale), inputs);
  }

  private JitOperation(
      JitOperator op,
      JitErrorPolicy errorPolicy,
      Integer targetScale,
      AstExpression... inputs) {
    this.op = Objects.requireNonNull(op, "op is null");
    this.errorPolicy = Objects.requireNonNull(errorPolicy, "errorPolicy is null");
    this.inputs = Objects.requireNonNull(inputs, "inputs is null").clone();
    this.targetScale = targetScale;
    if (this.inputs.length != op.getArity()) {
      throw new IllegalArgumentException(
          op + " requires " + op.getArity() + " inputs, found " + this.inputs.length);
    }
    if (!op.isFallible() && errorPolicy == JitErrorPolicy.NULLIFY) {
      throw new IllegalArgumentException(op + " cannot nullify errors");
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
        errorPolicy.getSerializedSize() +
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
    errorPolicy.serialize(bb);
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
    if (errorPolicy != JitErrorPolicy.PROPAGATE) {
      ret.append("[").append(errorPolicy).append("]");
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
