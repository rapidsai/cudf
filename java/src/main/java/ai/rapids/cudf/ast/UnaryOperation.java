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
}
