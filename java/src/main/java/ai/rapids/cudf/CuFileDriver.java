/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Represents a cuFile driver.
 */
final class CuFileDriver implements AutoCloseable {
  private final CuFileResourceCleaner cleaner;

  CuFileDriver() {
    cleaner = new CuFileResourceCleaner(create(), CuFileDriver::destroy);
    MemoryCleaner.register(this, cleaner);
  }

  @Override
  public void close() {
    cleaner.close(this);
  }

  private static native long create();

  private static native void destroy(long pointer);
}
