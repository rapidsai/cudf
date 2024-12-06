/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
