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
import java.nio.ByteOrder;

/** Base class of every node in an AST */
public abstract class AstExpression {
  /**
   * Enumeration for the types of AST nodes that can appear in a serialized AST.
   * NOTE: This must be kept in sync with the `jni_serialized_node_type` in CompiledExpression.cpp!
   */
  protected enum ExpressionType {
    VALID_LITERAL(0),
    NULL_LITERAL(1),
    COLUMN_REFERENCE(2),
    UNARY_EXPRESSION(3),
    BINARY_EXPRESSION(4);

    private final byte nativeId;

    ExpressionType(int nativeId) {
      this.nativeId = (byte) nativeId;
      assert this.nativeId == nativeId;
    }

    /** Get the size in bytes to serialize this node type */
    int getSerializedSize() {
      return Byte.BYTES;
    }

    /** Serialize this node type to the specified buffer */
    void serialize(ByteBuffer bb) {
      bb.put(nativeId);
    }
  }

  public CompiledExpression compile() {
    int size = getSerializedSize();
    ByteBuffer bb = ByteBuffer.allocate(size);
    bb.order(ByteOrder.nativeOrder());
    serialize(bb);
    return new CompiledExpression(bb.array());
  }

  /** Get the size in bytes of the serialized form of this node and all child nodes */
  abstract int getSerializedSize();

  /**
   * Serialize this node and all child nodes.
   * @param bb buffer to receive the serialized data
   */
  abstract void serialize(ByteBuffer bb);
}
