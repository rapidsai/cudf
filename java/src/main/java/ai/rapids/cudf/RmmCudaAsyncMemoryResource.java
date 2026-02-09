/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * A device memory resource that uses `cudaMallocAsync` and `cudaFreeAsync` for allocation and
 * deallocation.
 */
public class RmmCudaAsyncMemoryResource implements RmmDeviceMemoryResource {
  private final long releaseThreshold;
  private final long size;
  private long handle = 0;

  /**
   * Create a new async memory resource
   * @param size the initial size of the pool
   * @param releaseThreshold size in bytes for when memory is released back to cuda
   */
  public RmmCudaAsyncMemoryResource(long size, long releaseThreshold) {
    this(size, releaseThreshold, false);
  }

  /**
   * Create a new async memory resource
   * @param size the initial size of the pool
   * @param releaseThreshold size in bytes for when memory is released back to cuda
   * @param fabric if true request peer read+write accessible fabric handles when
   *        creating the pool
   */
  public RmmCudaAsyncMemoryResource(long size, long releaseThreshold, boolean fabric) {
    this.size = size;
    this.releaseThreshold = releaseThreshold;
    handle = Rmm.newCudaAsyncMemoryResource(size, releaseThreshold, fabric);
  }

  @Override
  public long getHandle() {
    return handle;
  }

  public long getSize() {
    return size;
  }

  @Override
  public void close() {
    if (handle != 0) {
      Rmm.releaseCudaAsyncMemoryResource(handle);
      handle = 0;
    }
  }

  @Override
  public String toString() {
    return Long.toHexString(getHandle()) + "/ASYNC(" + size + ", " + releaseThreshold + ")";
  }
}
