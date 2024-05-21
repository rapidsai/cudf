/*
 *
 *  Copyright (c) 2019-2024, NVIDIA CORPORATION.
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


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This is the JNI interface to a rmm::pool_memory_resource<rmm::pinned_host_memory_resource>.
 */
public final class PinnedMemoryPool implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(PinnedMemoryPool.class);

  // These static fields should only ever be accessed when class-synchronized.
  // Do NOT use singleton_ directly!  Use the getSingleton accessor instead.
  private static volatile PinnedMemoryPool singleton_ = null;
  private static Future<PinnedMemoryPool> initFuture = null;
  private long poolHandle;
  private long poolSize;

  private static final class PinnedHostBufferCleaner extends MemoryBuffer.MemoryBufferCleaner {
    private long address;
    private final long origLength;

    PinnedHostBufferCleaner(long address, long length) {
      this.address = address;
      origLength = length;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = 0;
      if (address != -1) {
        origAddress = address;
        try {
          PinnedMemoryPool.freeInternal(address, origLength);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          address = -1;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A PINNED HOST BUFFER WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked pinned host buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return address == -1;
    }
  }

  private static PinnedMemoryPool getSingleton() {
    if (singleton_ == null) {
      if (initFuture == null) {
        return null;
      }

      synchronized (PinnedMemoryPool.class) {
        if (singleton_ == null) {
          try {
            singleton_ = initFuture.get();
          } catch (Exception e) {
            throw new RuntimeException("Error initializing pinned memory pool", e);
          }
          initFuture = null;
        }
      }
    }
    return singleton_;
  }

  private static void freeInternal(long address, long origLength) {
    Objects.requireNonNull(getSingleton()).free(address, origLength);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   * @note when using this method, the pinned pool will be shared with cuIO
   */
  public static synchronized void initialize(long poolSize) {
    initialize(poolSize, -1, true);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   * @param gpuId    gpu id to set to get memory pool from, -1 means to use default
   * @note when using this method, the pinned pool will be shared with cuIO
   */
  public static synchronized void initialize(long poolSize, int gpuId) {
    initialize(poolSize, gpuId, true);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   * @param gpuId    gpu id to set to get memory pool from, -1 means to use default
   * @param setCuioHostMemoryResource true if this pinned pool should be used by cuIO for host memory
   */
  public static synchronized void initialize(long poolSize, int gpuId, boolean setCuioHostMemoryResource) {
    if (isInitialized()) {
      throw new IllegalStateException("Can only initialize the pool once.");
    }
    ExecutorService initService = Executors.newSingleThreadExecutor(runnable -> {
      Thread t = new Thread(runnable, "pinned pool init");
      t.setDaemon(true);
      return t;
    });
    initFuture = initService.submit(() -> new PinnedMemoryPool(poolSize, gpuId, setCuioHostMemoryResource));
    initService.shutdown();
  }

  /**
   * Check if the pool has been initialized or not.
   */
  public static boolean isInitialized() {
    return getSingleton() != null;
  }

  /**
   * Shut down the RMM pool_memory_resource, nulling out our reference. Any allocation
   * or free that is in flight will fail after this.
   */
  public static synchronized void shutdown() {
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      pool.close();
      pool = null;
    }
    initFuture = null;
    singleton_ = null;
  }

  /**
   * Factory method to create a pinned host memory buffer.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer or null if insufficient pinned memory
   */
  public static HostMemoryBuffer tryAllocate(long bytes) {
    HostMemoryBuffer result = null;
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      result = pool.tryAllocateInternal(bytes);
    }
    return result;
  }

  /**
   * Factory method to create a host buffer but preferably pointing to pinned memory.
   * It is not guaranteed that the returned buffer will be pointer to pinned memory.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes, HostMemoryAllocator hostMemoryAllocator) {
    HostMemoryBuffer result = tryAllocate(bytes);
    if (result == null) {
      result = hostMemoryAllocator.allocate(bytes, false);
    }
    return result;
  }

  /**
   * Factory method to create a host buffer but preferably pointing to pinned memory.
   * It is not guaranteed that the returned buffer will be pointer to pinned memory.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes) {
    return allocate(bytes, DefaultHostMemoryAllocator.get());
  }

  /**
   * Get the number of bytes that the pinned memory pool was allocated with.
   */
  public static long getTotalPoolSizeBytes() {
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      return pool.poolSize;
    }
    return 0;
  }

  private PinnedMemoryPool(long poolSize, int gpuId, boolean setCuioHostMemoryResource) {
    if (gpuId > -1) {
      // set the gpu device to use
      Cuda.setDevice(gpuId);
      Cuda.freeZero();
    }
    this.poolHandle = Rmm.newPinnedPoolMemoryResource(poolSize, poolSize);
    if (setCuioHostMemoryResource) {
      Rmm.setCuioPinnedPoolMemoryResource(this.poolHandle);
    }
    this.poolSize = poolSize;
  }

  @Override
  public void close() {
    Rmm.releasePinnedPoolMemoryResource(this.poolHandle);
    this.poolHandle = -1;
  }

  /**
   * This makes an attempt to allocate pinned memory, and if the pinned memory allocation fails
   * it will return null, instead of throw.
   */
  private synchronized HostMemoryBuffer tryAllocateInternal(long bytes) {
    long allocated = Rmm.allocFromPinnedPool(this.poolHandle, bytes);
    if (allocated == -1) {
      return null;
    } else {
      return new HostMemoryBuffer(allocated, bytes,
              new PinnedHostBufferCleaner(allocated, bytes));
    }
  }

  private synchronized void free(long address, long size) {
    Rmm.freeFromPinnedPool(this.poolHandle, address, size);
  }

  /**
   * Sets the size of the cuDF default pinned pool.
   *
   * @note This has to be called before cuDF functions are executed.
   *
   * @param size initial and maximum size for the cuDF default pinned pool.
   *        Pass size=0 to disable the default pool.
   */
  public static synchronized void configureDefaultCudfPinnedPoolSize(long size) {
    Rmm.configureDefaultCudfPinnedPoolSize(size);
  }

}
