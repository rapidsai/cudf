/*
 *
 *  Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import ai.rapids.cudf.nvcomp.BatchedLZ4Decompressor;
import ai.rapids.cudf.nvcomp.Decompressor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * ColumnVectors may store data off heap, and because of complicated processing the life time of
 * an individual vector can vary a lot.  Typically a java finalizer could be used for this but
 * they can cause a number of performance issues related to gc, and in some cases may effectively
 * leak resources if the heap is large and GC's end up being delayed.
 * <p>
 * To address these issues the primary way to releasing the resources of a ColumnVector that is
 * stored off of the java heap should be through reference counting. Because memory leaks are
 * really bad for long lived daemons this is intended to be a backup.
 * <p>
 * When a ColumnVector first allocates off heap resources it should register itself with this
 * along with a Cleaner instance. The Cleaner instance should have no direct links to the
 * ColumnVector that would prevent the ColumnVector from being garbage collected. This will
 * use WeakReferences internally to know when the resources have been leaked.
 * A ColumnVector may keep a reference to the Cleaner instance and either update it as new
 * resources are allocated or use it to release the resources it is holding.  Once the
 * ColumnVector's reference count reaches 0 and the resources are released. At some point
 * later the Cleaner itself will be released.
 */
public final class MemoryCleaner {
  private static final boolean REF_COUNT_DEBUG = Boolean.getBoolean("ai.rapids.refcount.debug");
  private static final Logger log = LoggerFactory.getLogger(MemoryCleaner.class);
  private static final AtomicLong idGen = new AtomicLong(0);

  /**
   * API that can be used to clean up the resources for a vector, even if there was a leak
   */
  public static abstract class Cleaner {
    private final List<RefCountDebugItem> refCountDebug;
    public final long id = idGen.incrementAndGet();
    private boolean leakExpected = false;

    public Cleaner() {
      if (REF_COUNT_DEBUG) {
        refCountDebug = new LinkedList<>();
      } else {
        refCountDebug = null;
      }
    }

    public final void addRef() {
      if (REF_COUNT_DEBUG && refCountDebug != null) {
        refCountDebug.add(new MemoryCleaner.RefCountDebugItem("INC"));
      }
    }

    public final void delRef() {
      if (REF_COUNT_DEBUG && refCountDebug != null) {
        refCountDebug.add(new MemoryCleaner.RefCountDebugItem("DEC"));
      }
    }

    public final void logRefCountDebug(String message) {
      if (REF_COUNT_DEBUG && refCountDebug != null) {
        log.error("{} (ID: {}): {}", message, id, MemoryCleaner.stringJoin("\n", refCountDebug));
      }
    }

    /**
     * Clean up any resources not previously released.
     * @param logErrorIfNotClean if true we should log a leak unless it is expected.
     * @return true if resources were cleaned up else false.
     */
    public final boolean clean(boolean logErrorIfNotClean) {
      return cleanImpl(logErrorIfNotClean && !leakExpected);
    }

    /**
     * Return true if a leak is expected for this object else false.
     */
    public final boolean isLeakExpected() {
      return leakExpected;
    }

    /**
     * Clean up any resources not previously released.
     * @param logErrorIfNotClean if true and there are resources to clean up a leak has happened
     *                           so log it.
     * @return true if resources were cleaned up else false.
     */
    protected abstract boolean cleanImpl(boolean logErrorIfNotClean);

    public void noWarnLeakExpected() {
      leakExpected = true;
    }

    /**
     * Check if the underlying memory has been cleaned up or not.
     * @return true this is clean else false.
     */
    public abstract boolean isClean();
  }

  static final AtomicLong leakCount = new AtomicLong();
  private static final Set<CleanerWeakReference> all =
      Collections.newSetFromMap(new ConcurrentHashMap()); // We want to be thread safe
  private static final ReferenceQueue<?> collected = new ReferenceQueue<>();

  private static class CleanerWeakReference<T> extends WeakReference<T> {

    private final Cleaner cleaner;
    final boolean isRmmBlocker;

    public CleanerWeakReference(T orig, Cleaner cleaner, ReferenceQueue collected, boolean isRmmBlocker) {
      super(orig, collected);
      this.cleaner = cleaner;
      this.isRmmBlocker = isRmmBlocker;
    }

    public void clean() {
      if (cleaner.clean(true)) {
        leakCount.incrementAndGet();
      }
    }
  }

  /**
   * The default GPU as set by user threads.
   */
  private static volatile int defaultGpu = -1;

  /**
   * This should be called from RMM when it is initialized.
   */
  static void setDefaultGpu(int defaultGpuId) {
    defaultGpu = defaultGpuId;
  }

  private static final Thread t = new Thread(() -> {
    try {
      int currentGpuId = -1;
      while (true) {
        CleanerWeakReference next = (CleanerWeakReference)collected.remove(100);
        if (next != null) {
          try {
            if (currentGpuId != defaultGpu) {
              Cuda.setDevice(defaultGpu);
              currentGpuId = defaultGpu;
            }
          } catch (Throwable t) {
            log.error("ERROR TRYING TO SET GPU ID TO " + defaultGpu, t);
          }
          try {
            next.clean();
          } catch (Throwable t) {
            log.error("CAUGHT EXCEPTION WHILE TRYING TO CLEAN " + next, t);
          }
          all.remove(next);
        }
      }
    } catch (InterruptedException e) {
      // Ignored just exit
    }
  }, "Cleaner Thread");

  static {
    t.setDaemon(true);
    t.start();
    if (REF_COUNT_DEBUG) {
      // If we are debugging things do a best effort to check for leaks at the end
      Runtime.getRuntime().addShutdownHook(new Thread(() -> {
        System.gc();
        // Avoid issues on shutdown with the cleaner thread.
        t.interrupt();
        try {
          t.join(1000);
        } catch (InterruptedException e) {
          // Ignored
        }
        if (defaultGpu >= 0) {
          Cuda.setDevice(defaultGpu);
        }
        for (CleanerWeakReference cwr : all) {
          cwr.clean();
        }
      }));
    }
  }

  static void register(ColumnVector vec, Cleaner cleaner) {
    // It is now registered...
    all.add(new CleanerWeakReference(vec, cleaner, collected, true));
  }

  static void register(HostColumnVectorCore vec, Cleaner cleaner) {
    // It is now registered...
    all.add(new CleanerWeakReference(vec, cleaner, collected, false));
  }

  static void register(MemoryBuffer buf, Cleaner cleaner) {
    // It is now registered...
    all.add(new CleanerWeakReference(buf, cleaner, collected, buf instanceof BaseDeviceMemoryBuffer));
  }

  static void register(Cuda.Stream stream, Cleaner cleaner) {
    // It is now registered...
    all.add(new CleanerWeakReference(stream, cleaner, collected, false));
  }

  static void register(Cuda.Event event, Cleaner cleaner) {
    // It is now registered...
    all.add(new CleanerWeakReference(event, cleaner, collected, false));
  }

  public static void register(Decompressor.Metadata metadata, Cleaner cleaner) {
    // It is now registered...
    all.add(new CleanerWeakReference(metadata, cleaner, collected, false));
  }

  public static void register(BatchedLZ4Decompressor.BatchedMetadata metadata, Cleaner cleaner) {
    // It is now registered...
    all.add(new CleanerWeakReference(metadata, cleaner, collected, false));
  }

  /**
   * This is not 100% perfect and we can still run into situations where RMM buffers were not
   * collected and this returns false because of thread race conditions. This is just a best effort.
   * @return true if there are rmm blockers else false.
   */
  static boolean bestEffortHasRmmBlockers() {
    return all.stream().anyMatch(cwr -> cwr.isRmmBlocker && !cwr.cleaner.isClean());
  }

  /**
   * Convert elements in it to a String and join them together. Only use for debug messages
   * where the code execution itself can be disabled as this is not fast.
   */
  private static <T> String stringJoin(String delim, Iterable<T> it) {
    return String.join(delim,
        StreamSupport.stream(it.spliterator(), false)
            .map((i) -> i.toString())
            .collect(Collectors.toList()));
  }

  /**
   * When debug is enabled holds information about inc and dec of ref count.
   */
  private static final class RefCountDebugItem {
    final StackTraceElement[] stackTrace;
    final long timeMs;
    final String op;

    public RefCountDebugItem(String op) {
      this.stackTrace = Thread.currentThread().getStackTrace();
      this.timeMs = System.currentTimeMillis();
      this.op = op;
    }

    public String toString() {
      Date date = new Date(timeMs);
      // Simple Date Format is horribly expensive only do this when debug is turned on!
      SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSSS z");
      return dateFormat.format(date) + ": " + op + "\n"
          + stringJoin("\n", Arrays.asList(stackTrace))
          + "\n";
    }
  }
}