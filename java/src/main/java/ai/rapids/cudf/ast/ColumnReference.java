/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

  @Override
  public String toString() {
    return tableSource.name() + " COLUMN(" + columnIndex + ")";
  }
}
