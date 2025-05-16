/*
 *
 *  Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

public class DefaultHostMemoryAllocator implements HostMemoryAllocator {
  private static volatile HostMemoryAllocator instance = new DefaultHostMemoryAllocator();

  /**
   * Retrieve current host memory allocator used by default if not passed directly to API
   *
   * @return current default HostMemoryAllocator implementation
   */
  public static HostMemoryAllocator get() {
    return instance;
  }

  /**
   * Sets a new default host memory allocator implementation by default.
   * @param hostMemoryAllocator the new allocator to use.
   */
  public static void set(HostMemoryAllocator hostMemoryAllocator) {
    instance = hostMemoryAllocator;
  }

  @Override
  public HostMemoryBuffer allocate(long bytes, boolean preferPinned) {
    if (preferPinned) {
      HostMemoryBuffer pinnedBuffer = PinnedMemoryPool.tryAllocate(bytes);
      if (pinnedBuffer != null) {
        return pinnedBuffer;
      }
    }
    return new HostMemoryBuffer(UnsafeMemoryAccessor.allocate(bytes), bytes);
  }

  @Override
  public HostMemoryBuffer allocate(long bytes) {
    return allocate(bytes, HostMemoryBuffer.defaultPreferPinned);
  }
}
