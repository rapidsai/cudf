/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import java.io.File;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * This is the binding class for rmm lib.
 */
public class Rmm {
  private static volatile boolean initialized = false;
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private enum LogLoc {
    NONE(0),
    FILE(1),
    STDOUT(2),
    STDERR(3);

    final int internalId;

    LogLoc(int internalId) {
      this.internalId = internalId;
    }
  }

  /**
   * What to send RMM alloc and free logs to.
   */
  public static class LogConf {
    private final File file;
    private final LogLoc loc;

    private LogConf(File file, LogLoc loc) {
      this.file = file;
      this.loc = loc;
    }
  }

  /**
   * Create a config that will write alloc/free logs to a file.
   */
  public static LogConf logTo(File location) {
    return new LogConf(location, LogLoc.FILE);
  }

  /**
   * Create a config that will write alloc/free logs to stdout.
   */
  public static LogConf logToStdout() {
    return new LogConf(null, LogLoc.STDOUT);
  }

  /**
   * Create a config that will write alloc/free logs to stderr.
   */
  public static LogConf logToStderr() {
    return new LogConf(null, LogLoc.STDERR);
  }

  /**
   * Initialize memory manager state and storage. This will always initialize
   * the CUDA context for the calling thread if it is not already set. The
   * caller is responsible for setting the desired CUDA device prior to this
   * call if a specific device is already set.
   * <p>NOTE: All cudf methods will set the chosen CUDA device in the CUDA
   * context of the calling thread after this returns.
   * @param allocationMode Allocation strategy to use. Bit set using
   *                       {@link RmmAllocationMode#CUDA_DEFAULT},
   *                       {@link RmmAllocationMode#POOL},
   *                       {@link RmmAllocationMode#ARENA},
   *                       {@link RmmAllocationMode#CUDA_ASYNC} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param logConf        How to do logging or null if you don't want to
   * @param poolSize       The initial pool size in bytes
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static synchronized void initialize(int allocationMode, LogConf logConf, long poolSize)
      throws RmmException {
    if (initialized) {
      throw new IllegalStateException("RMM is already initialized");
    }

    boolean isPool = (allocationMode & RmmAllocationMode.POOL) != 0;
    boolean isArena = (allocationMode & RmmAllocationMode.ARENA) != 0;
    boolean isAsync = (allocationMode & RmmAllocationMode.CUDA_ASYNC) != 0;
    boolean isManaged = (allocationMode & RmmAllocationMode.CUDA_MANAGED_MEMORY) != 0;

    if (isAsync && isManaged) {
      throw new IllegalArgumentException(
          "CUDA Unified Memory is not supported in CUDA_ASYNC allocation mode");
    }
    LogLoc loc = LogLoc.NONE;
    String path = null;
    if (logConf != null) {
      if (logConf.file != null) {
        path = logConf.file.getAbsolutePath();
      }
      loc = logConf.loc;
    }

    initializeInternal(allocationMode, loc.internalId, path, poolSize);
    MemoryCleaner.setDefaultGpu(Cuda.getDevice());
    initialized = true;
  }

  /**
   * Check if RMM has been initialized already or not.
   */
  public static boolean isInitialized() throws RmmException {
    return initialized;
  }

  /**
   * Return the amount of RMM memory allocated in bytes. Note that the result
   * may be less than the actual amount of allocated memory if underlying RMM
   * allocator decides to return more memory than what was requested. However
   * the result will always be a lower bound on the amount allocated.
   */
  public static native long getTotalBytesAllocated();

  /**
   * Returns the maximum amount of RMM memory (Bytes) outstanding during the
   * lifetime of the process.
   */
  public static native long getMaximumTotalBytesAllocated();

  /**
   * Resets a scoped maximum counter of RMM memory used to keep track of usage between
   * code sections while debugging.
   *
   * @param initialValue an initial value (in Bytes) to use for this scoped counter
   */
  public static void resetScopedMaximumBytesAllocated(long initialValue) {
    resetScopedMaximumBytesAllocatedInternal(initialValue);
  }

  /**
   * Resets a scoped maximum counter of RMM memory used to keep track of usage between
   * code sections while debugging.
   *
   * This resets the counter to 0 Bytes.
   */
  public static void resetScopedMaximumBytesAllocated() {
    resetScopedMaximumBytesAllocatedInternal(0L);
  }

  private static native void resetScopedMaximumBytesAllocatedInternal(long initialValue);

  /**
   * Returns the maximum amount of RMM memory (Bytes) outstanding since the last
   * `resetScopedMaximumOutstanding` call was issued (it is "scoped" because it's the
   * maximum amount seen since the last reset).
   *
   * If the memory used is net negative (for example if only frees happened since
   * reset, and we reset to 0), then result will be 0.
   *
   * If `resetScopedMaximumBytesAllocated` is never called, the scope is the whole
   * program and is equivalent to `getMaximumTotalBytesAllocated`.
   *
   * @return the scoped maximum bytes allocated
   */
  public static native long getScopedMaximumBytesAllocated();

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

  private static native void initializeInternal(int allocationMode, int logTo, String path,
      long poolSize) throws RmmException;

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
  public static synchronized void shutdown(long forceGCInterval, long maxWaitTime, TimeUnit units)
      throws RmmException {
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
    if (initialized) {
      shutdownInternal();
      initialized = false;
    }
  }

  private static native void shutdownInternal() throws RmmException;

  /**
   * Allocate device memory and return a pointer to device memory, using stream 0.
   * @param size   The size in bytes of the allocated memory region
   * @return Returned pointer to the allocated memory
   */
  public static DeviceMemoryBuffer alloc(long size) {
    return alloc(size, null);
  }

  /**
   * Allocate device memory and return a pointer to device memory.
   * @param size   The size in bytes of the allocated memory region
   * @param stream The stream in which to synchronize this command.
   * @return Returned pointer to the allocated memory
   */
  public static DeviceMemoryBuffer alloc(long size, Cuda.Stream stream) {
    long s = stream == null ? 0 : stream.getStream();
    return new DeviceMemoryBuffer(allocInternal(size, s), size, stream);
  }

  private static native long allocInternal(long size, long stream) throws RmmException;


  static native void free(long ptr, long length, long stream) throws RmmException;

  /**
   * Delete an rmm::device_buffer.
   */
  static native void freeDeviceBuffer(long rmmBufferAddress) throws RmmException;

  static native void setEventHandlerInternal(RmmEventHandler handler,
      long[] allocThresholds, long[] deallocThresholds) throws RmmException;

  /**
   * Allocate device memory using `cudaMalloc` and return a pointer to device memory.
   * @param size   The size in bytes of the allocated memory region
   * @param stream The stream in which to synchronize this command.
   * @return Returned pointer to the allocated memory
   */
  public static CudaMemoryBuffer allocCuda(long size, Cuda.Stream stream) {
    long s = stream == null ? 0 : stream.getStream();
    return new CudaMemoryBuffer(allocCudaInternal(size, s), size, stream);
  }

  private static native long allocCudaInternal(long size, long stream) throws RmmException;

  static native void freeCuda(long ptr, long length, long stream) throws RmmException;
}
