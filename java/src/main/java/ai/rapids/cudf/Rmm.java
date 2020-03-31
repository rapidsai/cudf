/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * This is the binding class for rmm lib.
 */
public class Rmm {
  private static volatile boolean defaultInitialized;
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Initialize memory manager state and storage.
   * @param allocationMode Allocation strategy to use. Bit set using
   *                       {@link RmmAllocationMode#CUDA_DEFAULT},
   *                       {@link RmmAllocationMode#POOL} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param enableLogging  Enable logging memory manager events
   * @param poolSize       The initial pool size in bytes
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static void initialize(int allocationMode, boolean enableLogging, long poolSize) {
    initialize(allocationMode, enableLogging, poolSize, Cuda.getDevice());
  }

  /**
   * Initialize memory manager state and storage.
   * @param allocationMode Allocation strategy to use. Bit set using
   *                       {@link RmmAllocationMode#CUDA_DEFAULT},
   *                       {@link RmmAllocationMode#POOL} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param enableLogging  Enable logging memory manager events
   * @param poolSize       The initial pool size in bytes
   * @param gpuId          The GPU that RMM should use.  You are still responsible for
   *                       setting this GPU yourself on each thread before calling RMM
   *                       this just sets it for any internal threads used.
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static void initialize(int allocationMode, boolean enableLogging, long poolSize, int gpuId)
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
    if (gpuId >= 0) {
      MemoryCleaner.setDefaultGpu(gpuId);
    }
  }

  /**
   * Sets the event handler to be called on RMM events (e.g.: allocation failure).
   * @param handler event handler to invoke on RMM events or null to clear an existing handler
   * @throws RmmException if an active handler is already set
   */
  public static void setEventHandler(RmmEventHandler handler) throws RmmException {
    long[] allocThresholds = (handler != null) ? sortThresholds(handler.getAllocThresholds()) : null;
    long[] deallocThresholds = (handler != null) ? sortThresholds(handler.getDeallocThresholds()) : null;
    setEventHandlerInternal(handler, allocThresholds, deallocThresholds);
  }

  /** Clears the active RMM event handler if one is set. */
  public static void clearEventHandler() throws RmmException {
    setEventHandlerInternal(null, null, null);
  }

  private static long[] sortThresholds(long[] thresholds) {
    if (thresholds == null) {
      return null;
    }
    long[] result = Arrays.copyOf(thresholds, thresholds.length);
    Arrays.sort(result);
    return result;
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

  private static native void initializeInternal(int allocationMode, boolean enableLogging,
      long poolSize) throws RmmException;

  /**
   * Check if RMM has been initialized already or not.
   */
  public static boolean isInitialized() throws RmmException {
    return !defaultInitialized && isInitializedInternal();
  }

  private static native boolean isInitializedInternal() throws RmmException;

  /**
   * Shut down any initialized RMM instance.  This should be used very rarely.  It does not need to
   * be used when shutting down your process because CUDA will handle releasing all of the
   * resources when your process exits.  This really should only be used if you want to turn off the
   * memory pool for some reasons.  As such we make an effort to be sure no resources have been
   * leaked before shutting down.  This may involve forcing a JVM GC to collect any leaked java
   * objects that still point to CUDA memory.  By default this will do a gc every 2 seconds and
   * wait for up to 4 seconds before throwing an RmmException if not all of the resources are freed.
   * @throws RmmException on any error. This includes if there are outstanding allocations that
   * could not be collected.
   */
  public static void shutdown() throws RmmException {
    shutdown(2, 4, TimeUnit.SECONDS);
  }

  /**
   * Shut down any initialized RMM instance.  This should be used very rarely.  It does not need to
   * be used when shutting down your process because CUDA will handle releasing all of the
   * resources when your process exits.  This really should only be used if you want to turn off the
   * memory pool for some reasons.  As such we make an effort to be sure no resources have been
   * leaked before shutting down.  This may involve forcing a JVM GC to collect any leaked java
   * objects that still point to CUDA memory.
   *
   * @param forceGCInterval how frequently should we force a JVM GC. This is just a recommendation
   *                        to the JVM to do a gc.
   * @param maxWaitTime the maximum amount of time to wait for all objects to be collected before
   *                    throwing an exception.
   * @param units the units for forceGcInterval and maxWaitTime.
   * @throws RmmException on any error. This includes if there are outstanding allocations that
   * could not be collected before maxWaitTime.
   */
  public static void shutdown(long forceGCInterval, long maxWaitTime, TimeUnit units)
      throws RmmException{
    long now = System.currentTimeMillis();
    final long endTime = now + units.toMillis(maxWaitTime);
    long nextGcTime = now;
    try {
      if (MemoryCleaner.bestEffortHasRmmBlockers()) {
        do {
          if (nextGcTime <= now) {
            System.gc();
            nextGcTime = nextGcTime + units.toMillis(forceGCInterval);
          }
          // Check if everything is ready about every 10 ms
          Thread.sleep(10);
          now = System.currentTimeMillis();
        } while (endTime > now && MemoryCleaner.bestEffortHasRmmBlockers());
      }
    } catch (InterruptedException e) {
      // Ignored
    }
    if (MemoryCleaner.bestEffortHasRmmBlockers()) {
      throw new RmmException("Could not shut down RMM there appear to be outstanding allocations");
    }
    shutdownInternal();
  }

  private native static void shutdownInternal() throws RmmException;

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
   */
  static native void free(long ptr, long stream) throws RmmException;

  /**
   * Delete an rmm::device_buffer.
   */
  static native void freeDeviceBuffer(long rmmBufferAddress) throws RmmException;

  static native void setEventHandlerInternal(RmmEventHandler handler,
      long[] allocThresholds, long[] deallocThresholds) throws RmmException;

  /**
   * If logging is enabled get the log as a String.
   */
  public static native String getLog() throws RmmException;
}
