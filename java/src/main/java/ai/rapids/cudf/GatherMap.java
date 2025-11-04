/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * This class tracks the data associated with a gather map, a buffer of INT32 elements that index
 * a source table and can be passed to a table gather operation.
 */
public class GatherMap implements AutoCloseable {
  private DeviceMemoryBuffer buffer;

  /**
   * Construct a gather map instance from a device buffer. The buffer length must be a multiple of
   * the {@link DType#INT32} size, as each row of the gather map is an INT32.
   * @param buffer device buffer backing the gather map data
   */
  public GatherMap(DeviceMemoryBuffer buffer) {
    if (buffer.getLength() % DType.INT32.getSizeInBytes() != 0) {
      throw new IllegalArgumentException("buffer length not a multiple of 4");
    }
    this.buffer = buffer;
  }

  @Override
  public String toString() {
    return "GatherMap(buffer=" + buffer + ")";
  }

  /** Return the number of rows in the gather map */
  public long getRowCount() {
    ensureOpen();
    return buffer.getLength() / 4;
  }

  /**
   * Create a column view that can be used to perform a gather operation. Note that the resulting
   * column view MUST NOT outlive the underlying device buffer within this instance!
   * @param startRow row offset where the resulting gather map will start
   * @param numRows number of rows in the resulting gather map
   * @return column view of gather map data
   */
  public ColumnView toColumnView(long startRow, int numRows) {
    ensureOpen();
    return ColumnView.fromDeviceBuffer(buffer, startRow * 4, DType.INT32, numRows);
  }

  /**
   * Release the underlying device buffer instance. After this is called, closing this instance
   * will not close the underlying device buffer. It is the responsibility of the caller to close
   * the returned device buffer.
   * @return device buffer backing gather map data or null if the buffer has already been released
   */
  public DeviceMemoryBuffer releaseBuffer() {
    DeviceMemoryBuffer result = buffer;
    buffer = null;
    return result;
  }

  /**
   * Get the address of the underlying device buffer.
   * This is to be used for JNI calls and with a lot of care.
   */
  public long getBufferAddress() {
    ensureOpen();
    return buffer.getAddress();
  }

  /**
   * Get the length of the underlying device buffer.
   * This is to be used for JNI calls and with a lot of care.
   * @return the length of the buffer in bytes.
   */
  public long getBufferLength() {
    ensureOpen();
    return buffer.getLength();
  }

  /** Close the device buffer backing the gather map data. */
  @Override
  public void close() {
    if (buffer != null) {
      buffer.close();
      buffer = null;
    }
  }

  private void ensureOpen() {
    if (buffer == null) {
      throw new IllegalStateException("instance is closed");
    }
    if (buffer.closed) {
      throw new IllegalStateException("buffer is closed");
    }
  }
}
