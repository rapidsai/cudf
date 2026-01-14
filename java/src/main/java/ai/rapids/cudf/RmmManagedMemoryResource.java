/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * A device memory resource that uses `cudaMallocManaged` and `cudaFreeManaged` for allocation and
 * deallocation.
 */
public class RmmManagedMemoryResource implements RmmDeviceMemoryResource {
  private long handle = 0;

  public RmmManagedMemoryResource() {
    handle = Rmm.newManagedMemoryResource();
  }
  @Override
  public long getHandle() {
    return handle;
  }

  @Override
  public void close() {
    if (handle != 0) {
      Rmm.releaseManagedMemoryResource(handle);
      handle = 0;
    }
  }

  @Override
  public String toString() {
    return Long.toHexString(getHandle()) + "/CUDA_MANAGED()";
  }
}
