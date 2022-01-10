/*
 *
 *  Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import java.util.Comparator;
import java.util.Iterator;
import java.util.Objects;
import java.util.Optional;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This provides a pool of pinned memory similar to what RMM does for device memory.
 */
public final class PinnedMemoryPool implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(PinnedMemoryPool.class);
  private static final long ALIGNMENT = 8;

  // These static fields should only ever be accessed when class-synchronized.
  // Do NOT use singleton_ directly!  Use the getSingleton accessor instead.
  private static volatile PinnedMemoryPool singleton_ = null;
  private static Future<PinnedMemoryPool> initFuture = null;

  private final long pinnedPoolBase;
  private final SortedSet<MemorySection> freeHeap = new TreeSet<>(new SortedByAddress());
  private int numAllocatedSections = 0;
  private long availableBytes;

  private static class SortedBySize implements Comparator<MemorySection> {
    @Override
    public int compare(MemorySection s0, MemorySection s1) {
      return Long.compare(s0.size, s1.size);
    }
  }

  private static class SortedByAddress implements Comparator<MemorySection> {
    @Override
    public int compare(MemorySection s0, MemorySection s1) {
      return Long.compare(s0.baseAddress, s1.baseAddress);
    }
  }

  private static class MemorySection {
    private long baseAddress;
    private long size;

    MemorySection(long baseAddress, long size) {
      this.baseAddress = baseAddress;
      this.size = size;
    }

    boolean canCombine(MemorySection other) {
      boolean ret = (other.baseAddress + other.size) == baseAddress ||
          (baseAddress + size) == other.baseAddress;
      log.trace("CAN {} COMBINE WITH {} ? {}", this, other, ret);
      return ret;
    }

    void combineWith(MemorySection other) {
      assert canCombine(other);
      log.trace("COMBINING {} AND {}", this, other);
      this.baseAddress = Math.min(baseAddress, other.baseAddress);
      this.size = other.size + this.size;
      log.trace("COMBINED TO {}\n", this);
    }

    MemorySection splitOff(long newSize) {
      assert this.size > newSize;
      MemorySection ret = new MemorySection(baseAddress, newSize);
      this.baseAddress += newSize;
      this.size -= newSize;
      return ret;
    }

    @Override
    public String toString() {
      return "PINNED: " + size + " bytes (0x" + Long.toHexString(baseAddress)
          + " to 0x" + Long.toHexString(baseAddress + size) + ")";
    }
  }

  private static final class PinnedHostBufferCleaner extends MemoryBuffer.MemoryBufferCleaner {
    private MemorySection section;
    private final long origLength;

    PinnedHostBufferCleaner(MemorySection section, long length) {
      this.section = section;
      origLength = length;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = 0;
      if (section != null) {
        origAddress = section.baseAddress;
        try {
          PinnedMemoryPool.freeInternal(section);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          section = null;
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
      return section == null;
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

  private static void freeInternal(MemorySection section) {
    Objects.requireNonNull(getSingleton()).free(section);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   */
  public static synchronized void initialize(long poolSize) {
    initialize(poolSize, -1);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   * @param gpuId    gpu id to set to get memory pool from, -1 means to use default
   */
  public static synchronized void initialize(long poolSize, int gpuId) {
    if (isInitialized()) {
      throw new IllegalStateException("Can only initialize the pool once.");
    }
    ExecutorService initService = Executors.newSingleThreadExecutor(runnable -> {
      Thread t = new Thread(runnable, "pinned pool init");
      t.setDaemon(true);
      return t;
    });
    initFuture = initService.submit(() -> new PinnedMemoryPool(poolSize, gpuId));
    initService.shutdown();
  }

  /**
   * Check if the pool has been initialized or not.
   */
  public static boolean isInitialized() {
    return getSingleton() != null;
  }

  /**
   * Shut down the pool of memory. If there are outstanding allocations this may fail.
   */
  public static synchronized void shutdown() {
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      pool.close();
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
  public static HostMemoryBuffer allocate(long bytes) {
    HostMemoryBuffer result = tryAllocate(bytes);
    if (result == null) {
      result = HostMemoryBuffer.allocate(bytes, false);
    }
    return result;
  }

  /**
   * Get the number of bytes free in the pinned memory pool.
   *
   * @return amount of free memory in bytes or 0 if the pool is not initialized
   */
  public static long getAvailableBytes() {
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      return pool.getAvailableBytesInternal();
    }
    return 0;
  }

  private PinnedMemoryPool(long poolSize, int gpuId) {
    if (gpuId > -1) {
      // set the gpu device to use
      Cuda.setDevice(gpuId);
      Cuda.freeZero();
    }
    this.pinnedPoolBase = Cuda.hostAllocPinned(poolSize);
    freeHeap.add(new MemorySection(pinnedPoolBase, poolSize));
    this.availableBytes = poolSize;
  }

  @Override
  public void close() {
    assert numAllocatedSections == 0;
    Cuda.freePinned(pinnedPoolBase);
  }

  private synchronized HostMemoryBuffer tryAllocateInternal(long bytes) {
    if (freeHeap.isEmpty()) {
      log.debug("No free pinned memory left");
      return null;
    }
    // Align the allocation
    long alignedBytes = ((bytes + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    Optional<MemorySection> firstFit = freeHeap.stream()
        .filter(section -> section.size >= alignedBytes)
        .findFirst();
    if (!firstFit.isPresent()) {
      if (log.isDebugEnabled()) {
        MemorySection largest = freeHeap.stream()
            .max(new SortedBySize())
            .orElse(new MemorySection(0, 0));
        log.debug("Insufficient pinned memory. {} needed, {} found", alignedBytes, largest.size);
      }
      return null;
    }
    MemorySection first = firstFit.get();
    log.debug("Allocating {}/{} bytes pinned from {} FREE COUNT {} OUTSTANDING COUNT {}",
        bytes, alignedBytes, first, freeHeap.size(), numAllocatedSections);
    freeHeap.remove(first);
    MemorySection allocated;
    if (first.size == alignedBytes) {
      allocated = first;
    } else {
      allocated = first.splitOff(alignedBytes);
      freeHeap.add(first);
    }
    numAllocatedSections++;
    availableBytes -= allocated.size;
    log.debug("Allocated {} free {} outstanding {}", allocated, freeHeap, numAllocatedSections);
    return new HostMemoryBuffer(allocated.baseAddress, bytes,
        new PinnedHostBufferCleaner(allocated, bytes));
  }

  private synchronized void free(MemorySection section) {
    log.debug("Freeing {} with {} outstanding {}", section, freeHeap, numAllocatedSections);
    availableBytes += section.size;
    Iterator<MemorySection> it = freeHeap.iterator();
    while(it.hasNext()) {
      MemorySection current = it.next();
      if (section.canCombine(current)) {
        it.remove();
        section.combineWith(current);
      }
    }
    freeHeap.add(section);
    numAllocatedSections--;
    log.debug("After freeing {} outstanding {}", freeHeap, numAllocatedSections);
  }

  private synchronized long getAvailableBytesInternal() {
    return this.availableBytes;
  }
}
