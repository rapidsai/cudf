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

/** A binary expression consisting of an operator and two operands. */
public class BinaryExpression extends Expression {
  private final AstOperator op;
  private final AstNode leftInput;
  private final AstNode rightInput;

  public BinaryExpression(AstOperator op, AstNode leftInput, AstNode rightInput) {
    if (!op.isBinaryOperator()) {
      throw new IllegalArgumentException(op + " is not a binary operator");
    }
    this.op = op;
    this.leftInput = leftInput;
    this.rightInput = rightInput;
  }

  @Override
  int getSerializedSize() {
    return NodeType.BINARY_EXPRESSION.getSerializedSize() +
        op.getSerializedSize() +
        leftInput.getSerializedSize() +
        rightInput.getSerializedSize();
  }

  @Override
  void serialize(ByteBuffer bb) {
    NodeType.BINARY_EXPRESSION.serialize(bb);
    op.serialize(bb);
    leftInput.serialize(bb);
    rightInput.serialize(bb);
  }
}
