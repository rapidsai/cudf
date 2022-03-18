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

/**
 * Enumeration of tables that can be referenced in an AST.
 * NOTE: This must be kept in sync with `jni_to_table_reference` code in CompiledExpression.cpp!
 */
public enum TableReference {
  LEFT(0),
  RIGHT(1);
  // OUTPUT is an AST implementation detail and should not appear in user-built expressions.

  private final byte nativeId;

  TableReference(int nativeId) {
    this.nativeId = (byte) nativeId;
    assert this.nativeId == nativeId;
  }

  /** Get the size in bytes to serialize this table reference */
  int getSerializedSize() {
    return Byte.BYTES;
  }

  /** Serialize this table reference to the specified buffer */
  void serialize(ByteBuffer bb) {
    bb.put(nativeId);
  }
}
