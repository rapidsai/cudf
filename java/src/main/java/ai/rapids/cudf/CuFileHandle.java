/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Represents a cuFile file handle.
 */
abstract class CuFileHandle implements AutoCloseable {
  private final CuFileResourceCleaner cleaner;

  static {
    CuFile.initialize();
  }

  protected CuFileHandle(long pointer) {
    cleaner = new CuFileResourceCleaner(pointer, CuFileHandle::destroy);
    MemoryCleaner.register(this, cleaner);
  }

  @Override
  public void close() {
    cleaner.close(this);
  }

  protected long getPointer() {
    return cleaner.getPointer();
  }

  private static native void destroy(long pointer);
}
