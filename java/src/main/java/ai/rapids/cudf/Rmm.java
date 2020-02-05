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

/**
 * This is the binding class for rmm lib.
 */
public class Rmm {
  private static volatile boolean defaultInitialized;
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * ---------------------------------------------------------------------------*
   * Initialize memory manager state and storage.
   * @param allocationMode Allocation strategy to use. Bit set using
   *                       {@link RmmAllocationMode#CUDA_DEFAULT},
   *                       {@link RmmAllocationMode#POOL} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param enableLogging  Enable logging memory manager events
   * @param poolSize       The initial pool size in bytes
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static void initialize(int allocationMode, boolean enableLogging, long poolSize)
      throws RmmException {
    if (defaultInitialized) {
      synchronized(Rmm.class) {
        if (defaultInitialized) {
          shutdown();
          defaultInitialized = false;
        }
      }
    }
    initializeInternal(allocationMode, enableLogging, poolSize);
  }

  /**
   * Initialize RMM in CUDA default mode.
   */
  static synchronized void defaultInitialize() {
    if (!defaultInitialized && !isInitializedInternal()) {
      initializeInternal(RmmAllocationMode.CUDA_DEFAULT, false, 0);
      defaultInitialized = true;
    }
  }

  private static native void initializeInternal(int allocationMode, boolean enableLogging, long poolSize)
      throws RmmException;

  /**
   * Check if RMM has been initialized already or not.
   */
  public static boolean isInitialized() throws RmmException {
    return !defaultInitialized && isInitializedInternal();
  }

  private static native boolean isInitializedInternal() throws RmmException;

  /**
   * Shut down any initialized rmm.
   */
  public static native void shutdown() throws RmmException;

  /**
   * ---------------------------------------------------------------------------*
   * Allocate memory and return a pointer to device memory.
   * If initialization has not occurred then RMM will be implicitly initialized
   * in CUDA default mode.
   * <p>
   * Mapping: RMM_ALLOC in rmm.h.
   * @param size   The size in bytes of the allocated memory region
   * @param stream The stream in which to synchronize this command
   * @return Returned pointer to the allocated memory
   */
  static native long alloc(long size, long stream) throws RmmException;


  /**
   * ---------------------------------------------------------------------------*
   * <p> Mapping: RMM_FREE in rmm.h
   * @param ptr    the pointer to memory location to be relinquished
   * @param stream the stream in which to synchronize this command
   * @throws RmmException
   */
  static native void free(long ptr, long stream) throws RmmException;

  /**
   * Delete an rmm::device_buffer.
   */
  static native void freeDeviceBuffer(long rmmBufferAddress) throws RmmException;

  /**
   * If logging is enabled get the log as a String.
   */
  public static native String getLog() throws RmmException;
}
