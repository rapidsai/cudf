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

  /**
   * What to send RMM alloc and free logs to.
   */
  public static class LogConf {
    private final File location;
    private final boolean stdout;
    private final boolean stderr;

    private LogConf(File location, boolean stdout, boolean stderr) {
      this.location = location;
      this.stdout = stdout;
      this.stderr = stderr;
    }
  }

  /**
   * Create a config that will write alloc/free logs to a file.
   */
  public static LogConf logTo(File location) {
    return new LogConf(location, false, false);
  }

  /**
   * Create a config that will write alloc/free logs to stdout.
   */
  public static LogConf logToStdout() {
    return new LogConf(null, true, false);
  }

  /**
   * Create a config that will write alloc/free logs to stderr.
   */
  public static LogConf logToStderr() {
    return new LogConf(null, false, true);
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
   * @param enableLogging  Enable logging memory manager events. If the environment variable
   *                       RMM_LOG_FILE is defined allocation logs will be written here,
   *                       otherwise they will be written to standard error.
   * @param poolSize       The initial pool size in bytes
   * @param gpuId          The GPU that RMM should use.  You are still responsible for
   *                       setting this GPU yourself on each thread before calling RMM
   *                       this just sets it for any internal threads used.
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static void initialize(int allocationMode, boolean enableLogging, long poolSize, int gpuId)
      throws RmmException {
    LogConf lc = null;
    if (enableLogging) {
      String f = System.getenv("RMM_LOG_FILE");
      if (f != null) {
        lc = logTo(new File(f));
      } else {
        lc = logToStderr();
      }
    }
    initialize(allocationMode, lc, poolSize, gpuId);
  }

  /**
   * Initialize memory manager state and storage.
   * @param allocationMode Allocation strategy to use. Bit set using
   *                       {@link RmmAllocationMode#CUDA_DEFAULT},
   *                       {@link RmmAllocationMode#POOL} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param logConf        How to do logging or null if you don't want to
   * @param poolSize       The initial pool size in bytes
   * @param gpuId          The GPU that RMM should use.  You are still responsible for
   *                       setting this GPU yourself on each thread before calling RMM
   *                       this just sets it for any internal threads used.
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static synchronized void initialize(int allocationMode, LogConf logConf, long poolSize, int gpuId)
      throws RmmException {
    if (initialized) {
      throw new IllegalStateException("RMM is already initialized");
    }
    int logTo = 0; // NONE
    String path = null;
    if (logConf != null) {
      if (logConf.location != null) {
        logTo = 1;
        path = logConf.location.getAbsolutePath();
      } else if (logConf.stdout) {
        logTo = 2;
      } else if (logConf.stderr) {
        logTo = 3;
      }
    }

    initializeInternal(allocationMode, logTo, path, poolSize);
    if (gpuId < 0) {
      gpuId = Cuda.getDevice();
    }
    MemoryCleaner.setDefaultGpu(gpuId);
    initialized = true;
  }

  /**
   * Check if RMM has been initialized already or not.
   */
  public static boolean isInitialized() throws RmmException {
    return initialized;
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
   * Small class used to ensure that we follow the rules of RMM allocation
   * and keep the size of the allocation around so we can free it without
   * any issues.  This keeps the stream around too, but that may change
   * in the future when we have better stream support, and the stream you
   * allocate something on is not always the one you want to synchronize against
   * when you free it.
   */
  public static class RmmBuff implements AutoCloseable {
    /**
     * The address of the allocation.
     */
    public final long ptr;
    /**
     * The length of the allocation.
     */
    public final long length;
    /**
     * The stream it was allocated on. This may change in the future once we have
     * better stream support.
     */
    public final long stream;
    private boolean closed = false;

    private RmmBuff(long ptr, long length, long stream) {
      this.ptr = ptr;
      this.length = length;
      this.stream = stream;
    }

    @Override
    public void close() {
      if (!closed) {
        free(ptr, length, stream);
        closed = true;
      }
    }
  }

  /**
   * Allocate device memory and return a pointer to device memory, using stream 0.
   * @param size   The size in bytes of the allocated memory region
   * @return Returned pointer to the allocated memory
   */
  public static RmmBuff alloc(long size) {
    return alloc(size, 0);
  }

  /**
   * Allocate device memory and return a pointer to device memory.
   * @param size   The size in bytes of the allocated memory region
   * @param stream The stream in which to synchronize this command
   * @return Returned pointer to the allocated memory
   */
  public static RmmBuff alloc(long size, long stream) {
    return new RmmBuff(allocInternal(size, stream), size, stream);
  }

  private static native long allocInternal(long size, long stream) throws RmmException;


  private static native void free(long ptr, long length, long stream) throws RmmException;

  /**
   * Delete an rmm::device_buffer.
   */
  static native void freeDeviceBuffer(long rmmBufferAddress) throws RmmException;

  static native void setEventHandlerInternal(RmmEventHandler handler,
      long[] allocThresholds, long[] deallocThresholds) throws RmmException;

  /**
   * Getting the results of alloc and free logging as a string is no longer supported.
   * @deprecated
   */
  @Deprecated
  public static String getLog() throws RmmException {
    throw new RuntimeException("In memory logging is no longer supported");
  }
}
