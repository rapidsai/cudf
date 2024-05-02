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
 * A device memory resource that will track some basic statistics about the memory usage.
 */
public class RmmTrackingResourceAdaptor<C extends RmmDeviceMemoryResource>
    extends RmmWrappingDeviceMemoryResource<C> {
  private long handle = 0;

  /**
   * Create a new tracking resource adaptor.
   * @param wrapped the memory resource to track allocations. This should not be reused.
   * @param alignment the alignment to apply.
   */
  public RmmTrackingResourceAdaptor(C wrapped, long alignment) {
    super(wrapped);
    handle = Rmm.newTrackingResourceAdaptor(wrapped.getHandle(), alignment);
  }

  @Override
  public long getHandle() {
    return handle;
  }

  public long getTotalBytesAllocated() {
    return Rmm.nativeGetTotalBytesAllocated(getHandle());
  }

  public long getMaxTotalBytesAllocated() {
    return Rmm.nativeGetMaxTotalBytesAllocated(getHandle());
  }

  public void resetScopedMaxTotalBytesAllocated(long initValue) {
    Rmm.nativeResetScopedMaxTotalBytesAllocated(getHandle(), initValue);
  }

  public long getScopedMaxTotalBytesAllocated() {
    return Rmm.nativeGetScopedMaxTotalBytesAllocated(getHandle());
  }

  @Override
  public void close() {
    if (handle != 0) {
      Rmm.releaseTrackingResourceAdaptor(handle);
      handle = 0;
    }
    super.close();
  }

  @Override
  public String toString() {
    return Long.toHexString(getHandle()) + "/TRACK(" + wrapped + ")";
  }
}
