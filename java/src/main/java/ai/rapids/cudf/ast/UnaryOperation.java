/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/** A unary operation consisting of an operator and an operand. */
public final class UnaryOperation extends AstExpression {
  private final UnaryOperator op;
  private final AstExpression input;

  public UnaryOperation(UnaryOperator op, AstExpression input) {
    this.op = op;
    this.input = input;
  }

  @Override
  int getSerializedSize() {
    return ExpressionType.UNARY_EXPRESSION.getSerializedSize() +
        op.getSerializedSize() +
        input.getSerializedSize();
  }

  @Override
  void serialize(ByteBuffer bb) {
    ExpressionType.UNARY_EXPRESSION.serialize(bb);
    op.serialize(bb);
    input.serialize(bb);
  }

  @Override
  public String toString() {
    return op + "(" + input + ")";
  }
}
