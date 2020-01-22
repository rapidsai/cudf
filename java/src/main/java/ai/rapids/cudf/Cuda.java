/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Cuda {
  // Defined in driver_types.h in cuda library.
  static final int CPU_DEVICE_ID = -1;
  private static final Logger log = LoggerFactory.getLogger(Cuda.class);
  private static Boolean isCompat = null;

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Mapping: cudaMemGetInfo(size_t *free, size_t *total)
   */
  public static native CudaMemInfo memGetInfo() throws CudaException;

  /**
   * Allocate pinned memory on the host.  This call takes a long time, but can really speed up
   * memory transfers.
   * @param size how much memory, in bytes, to allocate.
   * @return the address to the allocated memory.
   * @throws CudaException on any error.
   */
  static native long hostAllocPinned(long size) throws CudaException;

  /**
   * Free memory allocated with hostAllocPinned.
   * @param ptr the pointer returned by hostAllocPinned.
   * @throws CudaException on any error.
   */
  static native void freePinned(long ptr) throws CudaException;

  /**
   * Copies count bytes from the memory area pointed to by src to the memory area pointed to by
   * dst.
   * Calling cudaMemcpy() with dst and src pointers that do not
   * match the direction of the copy results in an undefined behavior.
   * @param dst   - Destination memory address
   * @param src   - Source memory address
   * @param count - Size in bytes to copy
   * @param kind  - Type of transfer. {@link CudaMemcpyKind}
   */
  static void memcpy(long dst, long src, long count, CudaMemcpyKind kind) {
    memcpy(dst, src, count, kind.getValue());
  }

  private static native void memcpy(long dst, long src, long count, int kind) throws CudaException;

  /**
   * Sets count bytes starting at the memory area pointed to by dst, with value.
   * @param dst   - Destination memory address
   * @param value - Byte value to set dst with
   * @param count - Size in bytes to set
   */
  public static native void memset(long dst, byte value, long count) throws CudaException;

  /**
   * Get the id of the current device.
   * @return the id of the current device
   * @throws CudaException on any error
   */
  public static native int getDevice() throws CudaException;

  /**
   * Set the id of the current device.
   * Note this is relative to CUDA_SET_VISIBLE_DEVICES, e.g. if
   * CUDA_SET_VISIBLE_DEVICES=1,0, and you call setDevice(0), you will get device 1.
   * @throws CudaException on any error
   */
  public static native void setDevice(int device) throws CudaException;

  /**
   * Calls cudaFree(0). This can be used to initialize the GPU after a setDevice()
   * @throws CudaException on any error
   */
  public static native void freeZero() throws CudaException;

  /**
   * This should only be used for tests, to enable or disable tests if the current environment
   * is not compatible with this version of the library.  Currently it only does some very
   * basic checks, but these may be expanded in the future depending on needs.
   * @return true if it is compatible else false.
   */
  public static synchronized boolean isEnvCompatibleForTesting() {
    if (isCompat == null) {
      if (NativeDepsLoader.libraryLoaded()) {
        try {
          int device = getDevice();
          if (device >= 0) {
            isCompat = true;
            return isCompat;
          }
        } catch (Throwable e) {
          log.error("Error trying to detect device", e);
        }
      }
      isCompat = false;
    }
    return isCompat;
  }
}
