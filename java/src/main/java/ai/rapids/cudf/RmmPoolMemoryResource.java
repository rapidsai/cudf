/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
