/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
 * Represents a cuFile buffer.
 */
public final class CuFileBuffer implements AutoCloseable {
  private static final int ALIGNMENT = 4096;

  private final long pointer;

  /**
   * Construct a new cuFile buffer.
   *
   * @param buffer The device memory buffer used for the cuFile buffer.
   * @param registerBuffer If true, register the cuFile buffer.
   */
  public CuFileBuffer(BaseDeviceMemoryBuffer buffer, boolean registerBuffer) {
    if (registerBuffer && !isAligned(buffer)) {
      throw new IllegalArgumentException(
          "To register a cuFile buffer, its length must be a multiple of " + ALIGNMENT);
    }
    pointer = create(buffer.address, buffer.length, registerBuffer);
  }

  @Override
  public void close() {
    destroy(pointer);
  }

  long getPointer() {
    return pointer;
  }

  private boolean isAligned(BaseDeviceMemoryBuffer buffer) {
    return buffer.length % ALIGNMENT == 0;
  }

  private static native long create(long address, long length, boolean registerBuffer);

  private static native void destroy(long pointer);
}
