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

/** A reference to a column in an input table. */
public final class ColumnReference extends AstExpression {
  private final int columnIndex;
  private final TableReference tableSource;

  /** Construct a column reference to either the only or leftmost input table */
  public ColumnReference(int columnIndex) {
    this(columnIndex, TableReference.LEFT);
  }

  /** Construct a column reference to the specified column index in the specified table */
  public ColumnReference(int columnIndex, TableReference tableSource) {
    this.columnIndex = columnIndex;
    this.tableSource = tableSource;
  }

  @Override
  int getSerializedSize() {
    // node type + table ref + column index
    return ExpressionType.COLUMN_REFERENCE.getSerializedSize() +
        tableSource.getSerializedSize() +
        Integer.BYTES;
  }

  @Override
  void serialize(ByteBuffer bb) {
    ExpressionType.COLUMN_REFERENCE.serialize(bb);
    tableSource.serialize(bb);
    bb.putInt(columnIndex);
  }
}
