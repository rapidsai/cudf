/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.nio.ByteBuffer;

/**
 * Metadata for a table that is backed by a single contiguous device buffer.
 */
public final class PackedColumnMetadata implements AutoCloseable {
  private long metadataHandle = 0;
  private ByteBuffer metadataBuffer = null;

  // This method is invoked by JNI
  static PackedColumnMetadata fromPackedColumnMeta(long metadataHandle) {
    return new PackedColumnMetadata(metadataHandle);
  }

  /**
   * Construct the PackedColumnMetadata instance given a metadata handle.
   * @param metadataHandle address of the cudf packed_table host-based metadata instance
   */
  PackedColumnMetadata(long metadataHandle) {
    this.metadataHandle = metadataHandle;
  }

  /**
   * Get the byte buffer containing the host metadata describing the schema and layout of the
   * contiguous table.
   * <p>
   * NOTE: This is a direct byte buffer that is backed by the underlying native metadata instance
   *       and therefore is only valid to be used while this PackedColumnMetadata instance is valid.
   *       Attempts to cache and access the resulting buffer after this instance has been destroyed
   *       will result in undefined behavior including the possibility of segmentation faults
   *       or data corruption.
   */
  public ByteBuffer getMetadataDirectBuffer() {
    if (metadataBuffer == null) {
      metadataBuffer = createMetadataDirectBuffer(metadataHandle);
    }
    return metadataBuffer.asReadOnlyBuffer();
  }

  /** Close the PackedColumnMetadata instance and its underlying resources. */
  @Override
  public void close() {
    if (metadataHandle != 0) {
      closeMetadata(metadataHandle);
      metadataHandle = 0;
    }
  }

  // create a DirectByteBuffer for the packed metadata
  private static native ByteBuffer createMetadataDirectBuffer(long metadataHandle);

  // release the native metadata resources for a packed table
  private static native void closeMetadata(long metadataHandle);
}
