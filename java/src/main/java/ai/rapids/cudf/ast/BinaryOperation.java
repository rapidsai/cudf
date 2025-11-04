/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/** A binary operation consisting of an operator and two operands. */
public class BinaryOperation extends AstExpression {
  private final BinaryOperator op;
  private final AstExpression leftInput;
  private final AstExpression rightInput;

  public BinaryOperation(BinaryOperator op, AstExpression leftInput, AstExpression rightInput) {
    this.op = op;
    this.leftInput = leftInput;
    this.rightInput = rightInput;
  }

  @Override
  int getSerializedSize() {
    return ExpressionType.BINARY_EXPRESSION.getSerializedSize() +
        op.getSerializedSize() +
        leftInput.getSerializedSize() +
        rightInput.getSerializedSize();
  }

  @Override
  void serialize(ByteBuffer bb) {
    ExpressionType.BINARY_EXPRESSION.serialize(bb);
    op.serialize(bb);
    leftInput.serialize(bb);
    rightInput.serialize(bb);
  }

  @Override
  public String toString() {
    return "(" + leftInput + " " + op + " " + rightInput + ")";
  }
}
