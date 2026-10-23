/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Owning result of a status-aware Variant extraction or cast.
 *
 * <p>The value and status columns have the same row count. The status column is a non-nullable
 * {@link DType#UINT8} column containing IDs defined by {@link VariantOperationStatus}. Both columns
 * are owned by this result and remain valid until it is closed. Callers must not close the returned
 * columns separately unless they first increment their reference counts with
 * {@link ColumnVector#incRefCount()}. A retained reference must then be closed separately.
 * Closing this result more than once has no effect.
 */
@Experimental
public final class VariantOperationResult implements AutoCloseable {
  private static final int VALUE_COLUMN_INDEX = 0;
  private static final int STATUS_COLUMN_INDEX = 1;

  private final Table table;

  VariantOperationResult(long[] columnHandles) {
    if (columnHandles == null || columnHandles.length != 2) {
      throw new IllegalArgumentException("a Variant operation result must contain two columns");
    }
    table = new Table(columnHandles);
  }

  /**
   * Return the operation value column.
   *
   * @return the value column, owned by this result
   */
  public ColumnVector getValue() {
    return table.getColumn(VALUE_COLUMN_INDEX);
  }

  /**
   * Return the non-nullable UINT8 operation status column.
   *
   * @return the status column, owned by this result
   */
  public ColumnVector getStatus() {
    return table.getColumn(STATUS_COLUMN_INDEX);
  }

  @Override
  public void close() {
    table.close();
  }
}
