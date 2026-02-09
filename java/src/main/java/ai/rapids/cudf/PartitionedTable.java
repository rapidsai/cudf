/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Class to provide a PartitionedTable
 */
public final class PartitionedTable implements AutoCloseable {
  private final Table table;
  private final int[] partitionsOffsets;

  /**
   * The package-private constructor is only called by the partition method in Table
   * .TableOperation.partition
   * @param table            - {@link Table} which contains the partitioned data
   * @param partitionOffsets - This param is used to populate the offsets into the returned table
   *                         where partitionOffsets[i] indicates the starting position of
   *                         partition 'i'
   */
  PartitionedTable(Table table, int[] partitionOffsets) {
    this.table = table;
    this.partitionsOffsets = partitionOffsets;
  }

  public Table getTable() {
    return table;
  }

  public ColumnVector getColumn(int index) {
    return table.getColumn(index);
  }

  public long getNumberOfColumns() {
    return table.getNumberOfColumns();
  }

  public long getRowCount() {
    return table.getRowCount();
  }

  @Override
  public void close() {
    table.close();
  }

  /**
   * This method returns the partitions on this table. partitionOffsets[i] indicates the
   * starting position of partition 'i' in the partitioned table. Size of the partitions can
   * be calculated by the next offset
   * Ex:
   * partitionOffsets[0, 12, 12, 49] indicates 4 partitions with the following sizes
   * partition[0] - 12
   * partition[1] - 0 (is empty)
   * partition[2] - 37
   * partition[3] has the remaining values of the table (N-49)
   */
  public int[] getPartitions() {
    return partitionsOffsets;
  }
}
