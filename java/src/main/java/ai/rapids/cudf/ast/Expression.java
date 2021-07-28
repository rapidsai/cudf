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

/** Base class of every AST expression. */
public abstract class Expression extends AstNode {
  public CompiledExpression compile() {
    int size = getSerializedSize();
    ByteBuffer bb = ByteBuffer.allocate(size);
    bb.order(ByteOrder.nativeOrder());
    serialize(bb);
    return new CompiledExpression(bb.array());
  }
}
