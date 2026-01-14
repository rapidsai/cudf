/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * A device memory resource that will pre-allocate a pool of resources and sub-allocate from this
 * pool to improve memory performance.
 */
public class RmmPoolMemoryResource<C extends RmmDeviceMemoryResource>
    extends RmmWrappingDeviceMemoryResource<C> {
  private long handle = 0;
  private final long initSize;
  private final long maxSize;

  /**
   * Create a new pooled memory resource taking ownership of the RmmDeviceMemoryResource that it is
   * wrapping.
   * @param wrapped the memory resource to use for the pool. This should not be reused.
   * @param initSize the size of the initial pool
   * @param maxSize the size of the maximum pool
   */
  public RmmPoolMemoryResource(C wrapped, long initSize, long maxSize) {
    super(wrapped);
    this.initSize = initSize;
    this.maxSize = maxSize;
    handle = Rmm.newPoolMemoryResource(wrapped.getHandle(), initSize, maxSize);
  }

  public long getMaxSize() {
    return maxSize;
  }

  @Override
  public long getHandle() {
    return handle;
  }

  @Override
  public void close() {
    if (handle != 0) {
      Rmm.releasePoolMemoryResource(handle);
      handle = 0;
    }
    super.close();
  }

  @Override
  public String toString() {
    return Long.toHexString(getHandle()) + "/POOL(" + wrapped + ")";
  }
}
