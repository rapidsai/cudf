/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Represents a cuFile file handle for reading.
 */
public final class CuFileReadHandle extends CuFileHandle {

  /**
   * Construct a reader using the specified file path.
   *
   * @param path The file path for reading.
   */
  public CuFileReadHandle(String path) {
    super(create(path));
  }

  /**
   * Read the file content into the specified cuFile buffer.
   *
   * @param buffer The cuFile buffer to store the content.
   * @param fileOffset The file offset from which to read.
   */
  public void read(CuFileBuffer buffer, long fileOffset) {
    readIntoBuffer(getPointer(), fileOffset, buffer.getPointer());
  }

  private static native long create(String path);

  private static native void readIntoBuffer(long file, long fileOffset, long buffer);
}
