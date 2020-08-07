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
   *                       {@link RmmAllocationMode#POOL} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param enableLogging  Enable logging memory manager events
   * @param poolSize       The initial pool size in bytes
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static void initialize(int allocationMode, boolean enableLogging, long poolSize)
      throws RmmException {
    initialize(allocationMode, enableLogging, poolSize, 0);
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
   *                       {@link RmmAllocationMode#POOL} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param enableLogging  Enable logging memory manager events
   * @param poolSize       The initial pool size in bytes
   * @param maxPoolSize    The maximum size the pool is allowed to grow. If the specified value
   *                       is <= 0 then the maximum pool size will not be artificially limited.
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static void initialize(int allocationMode, boolean enableLogging, long poolSize,
      long maxPoolSize) throws RmmException {
    LogConf lc = null;
    if (enableLogging) {
      String f = System.getenv("RMM_LOG_FILE");
      if (f != null) {
        lc = logTo(new File(f));
      } else {
        lc = logToStderr();
      }
    }
    initialize(allocationMode, lc, poolSize, maxPoolSize);
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
   *                       {@link RmmAllocationMode#POOL} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param logConf        How to do logging or null if you don't want to
   * @param poolSize       The initial pool size in bytes
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static synchronized void initialize(int allocationMode, LogConf logConf, long poolSize)
      throws RmmException {
    initialize(allocationMode, logConf, poolSize, 0);
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
   *                       {@link RmmAllocationMode#POOL} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param logConf        How to do logging or null if you don't want to
   * @param poolSize       The initial pool size in bytes
   * @param maxPoolSize    The maximum size the pool is allowed to grow. If the specified value
   *                       is <= 0 then the pool size will not be artificially limited.
   * @throws IllegalStateException if RMM has already been initialized
   * @throws IllegalArgumentException if a max pool size is specified but the allocation mode
   *                                  is not {@link RmmAllocationMode#POOL} or the maximum pool
   *                                  size is below the initial size.
   */
  public static synchronized void initialize(int allocationMode, LogConf logConf, long poolSize,
      long maxPoolSize) throws RmmException {
    if (initialized) {
      throw new IllegalStateException("RMM is already initialized");
    }
    if (maxPoolSize > 0) {
      if (allocationMode != RmmAllocationMode.POOL) {
        throw new IllegalArgumentException("Pool limit only supported in POOL allocation mode");
      }
      if (maxPoolSize < poolSize) {
        throw new IllegalArgumentException("Pool limit of " + maxPoolSize
            + " is less than initial pool size of " + poolSize);
      }
    }
    LogLoc loc = LogLoc.NONE;
    String path = null;
    if (logConf != null) {
      if (logConf.file != null) {
        path = logConf.file.getAbsolutePath();
      }
      loc = logConf.loc;
    }

    initializeInternal(allocationMode, loc.internalId, path, poolSize, maxPoolSize);
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
      long poolSize, long maxPoolSize) throws RmmException;

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
  static DeviceMemoryBuffer alloc(long size, Cuda.Stream stream) {
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
}
