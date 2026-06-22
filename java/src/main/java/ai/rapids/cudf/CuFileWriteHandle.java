/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Represents a cuFile file handle for reading.
 */
public final class CuFileWriteHandle extends CuFileHandle {

  /**
   * Construct a writer using the specified file path.
   *
   * @param path The file path for writing.
   */
  public CuFileWriteHandle(String path) {
    super(create(path));
  }

  /**
   * Write the specified cuFile buffer into the file.
   *
   * @param buffer The cuFile buffer to write from.
   * @param length The number of bytes to write.
   * @param fileOffset The starting file offset from which to write.
   */
  public void write(CuFileBuffer buffer, long length, long fileOffset) {
    writeFromBuffer(getPointer(), fileOffset, buffer.getPointer(), length);
  }

  /**
   * Append the specified cuFile buffer to the file.
   *
   * @param buffer The cuFile buffer to append from.
   * @param length The number of bytes to append.
   * @return The file offset from which the buffer was appended.
   */
  public long append(CuFileBuffer buffer, long length) {
    return appendFromBuffer(getPointer(), buffer.getPointer(), length);
  }

  private static native long create(String path);

  private static native void writeFromBuffer(long file, long fileOffset, long buffer, long length);

  private static native long appendFromBuffer(long file, long buffer, long length);
}
