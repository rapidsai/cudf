/*
 *
 *  Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

import java.nio.ByteBuffer;

/**
 * A table that is backed by a single contiguous device buffer. This makes transfers of the data
 * much simpler.
 */
public final class ContiguousTable implements AutoCloseable {
  private long metadataHandle = 0;
  private Table table = null;
  private DeviceMemoryBuffer buffer;
  private ByteBuffer metadataBuffer = null;

  // This method is invoked by JNI
  static ContiguousTable fromPackedTable(long metadataHandle,
                                         long dataAddress,
                                         long dataLength,
                                         long rmmBufferAddress) {
    DeviceMemoryBuffer buffer = DeviceMemoryBuffer.fromRmm(dataAddress, dataLength, rmmBufferAddress);
    return new ContiguousTable(metadataHandle, buffer);
  }

  /** Construct a contiguous table instance given a table and the device buffer backing it. */
  ContiguousTable(Table table, DeviceMemoryBuffer buffer) {
    this.table = table;
    this.buffer = buffer;
  }

  /**
   * Construct a contiguous table
   * @param metadataHandle address of the cudf packed_table host-based metadata instance
   * @param buffer buffer containing the packed table data
   */
  ContiguousTable(long metadataHandle, DeviceMemoryBuffer buffer) {
    this.metadataHandle = metadataHandle;
    this.buffer = buffer;
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
    if (metadataBuffer == null) {
      metadataBuffer = createMetadataDirectBuffer(metadataHandle);
    }
    return metadataBuffer;
  }

  /** Close the contiguous table instance and its underlying resources. */
  @Override
  public void close() {
    if (metadataHandle != 0) {
      closeMetadata(metadataHandle);
      metadataHandle = 0;
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

  // create a DirectByteBuffer for the packed table metadata
  private static native ByteBuffer createMetadataDirectBuffer(long metadataHandle);

  // release the native metadata resources for a packed table
  private static native void closeMetadata(long metadataHandle);
}
