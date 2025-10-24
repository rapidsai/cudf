/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.nio.ByteBuffer;

/**
 * A table that is backed by a single contiguous device buffer. This makes transfers of the data
 * much simpler.
 */
public final class ContiguousTable implements AutoCloseable {
  private Table table = null;
  private DeviceMemoryBuffer buffer;
  private final long rowCount;
  private PackedColumnMetadata meta;
  private ByteBuffer metadataBuffer;

  // This method is invoked by JNI
  static ContiguousTable fromPackedTable(long metadataHandle,
                                         long dataAddress,
                                         long dataLength,
                                         long rmmBufferAddress,
                                         long rowCount) {
    DeviceMemoryBuffer buffer = DeviceMemoryBuffer.fromRmm(dataAddress, dataLength, rmmBufferAddress);
    return new ContiguousTable(metadataHandle, buffer, rowCount);
  }

  /** Construct a contiguous table instance given a table and the device buffer backing it. */
  ContiguousTable(Table table, DeviceMemoryBuffer buffer) {
    this.meta = new PackedColumnMetadata(createPackedMetadata(table.getNativeView(),
            buffer.getAddress(), buffer.getLength()));
    this.table = table;
    this.buffer = buffer;
    this.rowCount = table.getRowCount();
  }

  /**
   * Construct a contiguous table
   * @param metadataHandle address of the cudf packed_table host-based metadata instance
   * @param buffer buffer containing the packed table data
   * @param rowCount number of rows in the table
   */
  ContiguousTable(long metadataHandle, DeviceMemoryBuffer buffer, long rowCount) {
    this.meta = new PackedColumnMetadata(metadataHandle);
    this.buffer = buffer;
    this.rowCount = rowCount;
  }

  /**
   * Returns the number of rows in the table. This accessor avoids manifesting
   * the Table instance if only the row count is needed.
   */
  public long getRowCount() {
    return rowCount;
  }

  /** Get the table instance, reconstructing it from the metadata if necessary. */
  public synchronized Table getTable() {
    if (table == null) {
      table = Table.fromPackedTable(getMetadataDirectBuffer(), buffer);
    }
    return table;
  }

  /** Get the device buffer backing the contiguous table data. */
  public DeviceMemoryBuffer getBuffer() {
    return buffer;
  }

  /**
   * Get the byte buffer containing the host metadata describing the schema and layout of the
   * contiguous table.
   * <p>
   * NOTE: This is a direct byte buffer that is backed by the underlying native metadata instance
   *       and therefore is only valid to be used while this contiguous table instance is valid.
   *       Attempts to cache and access the resulting buffer after this instance has been destroyed
   *       will result in undefined behavior including the possibility of segmentation faults
   *       or data corruption.
   */
  public ByteBuffer getMetadataDirectBuffer() {
    return meta.getMetadataDirectBuffer();
  }

  /** Close the contiguous table instance and its underlying resources. */
  @Override
  public void close() {
    if (meta != null) {
      meta.close();
    }

    if (table != null) {
      table.close();
      table = null;
    }

    if (buffer != null) {
      buffer.close();
      buffer = null;
    }
  }

  // create packed metadata for a table backed by a single data buffer
  private static native long createPackedMetadata(long tableView, long dataAddress, long dataSize);

}
