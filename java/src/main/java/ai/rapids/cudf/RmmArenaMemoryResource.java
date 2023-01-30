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
 * pool to improve memory performance. This uses an algorithm to try and reduce fragmentation
 * much more than the RmmPoolMemoryResource does.
 */
public class RmmArenaMemoryResource<C extends RmmDeviceMemoryResource>
    extends RmmWrappingDeviceMemoryResource<C> {
  private final long size;
  private final boolean dumpLogOnFailure;
  private long handle = 0;


  /**
   * Create a new arena memory resource taking ownership of the RmmDeviceMemoryResource that it is
   * wrapping.
   * @param wrapped the memory resource to use for the pool. This should not be reused.
   * @param size the size of the pool
   * @param dumpLogOnFailure if true, dump memory log when running out of memory.
   */
  public RmmArenaMemoryResource(C wrapped, long size, boolean dumpLogOnFailure) {
    super(wrapped);
    this.size = size;
    this.dumpLogOnFailure = dumpLogOnFailure;
    handle = Rmm.newArenaMemoryResource(wrapped.getHandle(), size, dumpLogOnFailure);
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
      Rmm.releaseArenaMemoryResource(handle);
      handle = 0;
    }
    super.close();
  }

  @Override
  public String toString() {
    return Long.toHexString(getHandle()) + "/ARENA(" + wrapped +
        ", " + size + ", " + dumpLogOnFailure + ")";
  }
}
