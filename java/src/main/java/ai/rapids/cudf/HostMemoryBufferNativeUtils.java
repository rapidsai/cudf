/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
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

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Wrapper for {@link HostMemoryBuffer} native callbacks so that class avoids
 * loading the native libraries unless one if its methods requires it.
 */
class HostMemoryBufferNativeUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * This will turn an address into a ByteBuffer.  The buffer will NOT own the memory
   * so closing it has no impact on the underlying memory. It should never
   * be used if the corresponding HostMemoryBuffer is closed.
   */
  static native ByteBuffer wrapRangeInBuffer(long address, long len);

  /**
   * Memory map a portion of a local file
   * @param file path to the local file to be mapped
   * @param mode 0=read, 1=read+write
   * @param offset file offset where map starts. Must be a system page boundary.
   * @param len number of bytes to map
   * @return address of the memory-mapped region
   * @throws IOException I/O error during mapping
   */
  static native long mmap(String file, int mode, long offset, long len) throws IOException;

  /**
   * Unmap a memory region that was memory-mapped.
   * @param address address of the memory-mapped region
   * @param length size of the mapped region in bytes
   */
  static native void munmap(long address, long length);
}
