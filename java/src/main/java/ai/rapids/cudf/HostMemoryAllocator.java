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

public interface HostMemoryAllocator {

  /**
   * Allocate memory, but be sure to close the returned buffer to avoid memory leaks.
   * @param bytes size in bytes to allocate
   * @param preferPinned If set to true, the pinned memory pool will be used if possible with a
   *                    fallback to off-heap memory.  If set to false, the allocation will always
   *                    be from off-heap memory.
   * @return the newly created buffer
   */
  HostMemoryBuffer allocate(long bytes, boolean preferPinned);

  /**
   * Allocate memory, but be sure to close the returned buffer to avoid memory leaks. Pinned memory
   * for allocations preference is up to the implementor
   *
   * @param bytes size in bytes to allocate
   * @return the newly created buffer
   */
  HostMemoryBuffer allocate(long bytes);
}
