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

  private static class StreamCleaner extends MemoryCleaner.Cleaner {
    private long stream;

    StreamCleaner(long stream) {
      this.stream = stream;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = stream;
      if (stream != 0) {
        destroyStream(stream);
        stream = 0;
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A CUDA STREAM WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked stream");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return stream == 0;
    }
  }

  /** A class representing a CUDA stream */
  public static final class Stream implements AutoCloseable {
    private final StreamCleaner cleaner;
    boolean closed = false;
    private final long id;

    /**
     * Create a new CUDA stream
     * @param isNonBlocking whether stream should be non-blocking with respect to the default stream
     */
    public Stream(boolean isNonBlocking) {
      this.cleaner = new StreamCleaner(createStream(isNonBlocking));
      this.id = cleaner.id;
      MemoryCleaner.register(this, cleaner);
      cleaner.addRef();
    }

    private Stream() {
      // No cleaner for the default stream...
      this.cleaner = null;
      this.id = -1;
    }

    /**
     * Have this stream not execute new work until the work recorded in event completes.
     * @param event the event to wait on.
     */
    public void waitOn(Event event) {
      streamWaitEvent(getStream(), event.getEvent());
    }

    long getStream() {
      return cleaner == null ? 0 : cleaner.stream;
    }

    /**
     * Block the thread to wait until all pending work on this stream completes.  Note that this
     * does not follow any of the java threading standards.  Interrupt will not work to wake up
     * the thread.
     */
    public void sync() {
      streamSynchronize(getStream());
    }

    @Override
    public String toString() {
      return "CUDA STREAM (ID: " + id + " " + Long.toHexString(getStream()) + ")";
    }

    @Override
    public void close() {
      if (cleaner != null) {
        cleaner.delRef();
      }
      if (closed) {
        cleaner.logRefCountDebug("double free " + this);
        throw new IllegalStateException("Close called too many times " + this);
      }
      if (cleaner != null) {
        cleaner.clean(false);
        closed = true;
      }
    }
  }

  public static final Stream DEFAULT_STREAM = new Stream();

  private static class EventCleaner extends MemoryCleaner.Cleaner {
    private long event;

    EventCleaner(long event) {
      this.event = event;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = event;
      if (event != 0) {
        try {
          destroyEvent(event);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          event = 0;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A CUDA EVENT WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked event");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return event == 0;
    }
  }

  public static final class Event implements AutoCloseable {
    private final EventCleaner cleaner;
    boolean closed = false;

    /**
     * Create an event that is as fast as possible, timing is disabled and no blockingSync.
     */
    public Event() {
      this(false, false);
    }

    /**
     * Create an event to be used for CUDA synchronization.
     * @param enableTiming true if the event should record timing information.
     * @param blockingSync true if event should use blocking synchronization.
     *                     A host thread that calls sync() to wait on an event created with this
     *                     flag will block until the event actually completes.
     */
    public Event(boolean enableTiming, boolean blockingSync) {
      this.cleaner = new EventCleaner(createEvent(enableTiming, blockingSync));
      MemoryCleaner.register(this, cleaner);
      cleaner.addRef();
    }

    long getEvent() {
      return cleaner.event;
    }

    /**
     * Check to see if the event has completed or not. This is the equivalent of cudaEventQuery.
     * @return true it has completed else false.
     */
    public boolean hasCompleted() {
      return eventQuery(getEvent());
    }

    /**
     * Captures the contents of stream at the time of this call. This event and stream must be on
     * the same device. Calls such as hasCompleted() or Stream.waitEvent() will then examine or wait for
     * completion of the work that was captured. Uses of stream after this call do not modify event.
     * @param stream the stream to record the state of.
     */
    public void record(Stream stream) {
      eventRecord(getEvent(), stream.getStream());
    }

    /**
     * Captures the contents of stream 0 at the time of this call.
     */
    public void record() {
      record(DEFAULT_STREAM);
    }

    /**
     * Block the thread to wait for the event to complete.  Note that this does not follow any of
     * the java threading standards.  Interrupt will not work to wake up the thread.
     */
    public void sync() {
      eventSynchronize(getEvent());
    }

    @Override
    public String toString() {
      return "CUDA EVENT (ID: " + cleaner.id + " " + Long.toHexString(getEvent()) + ")";
    }

    @Override
    public void close() {
      cleaner.delRef();
      if (closed) {
        cleaner.logRefCountDebug("double free " + this);
        throw new IllegalStateException("Close called too many times " + this);
      }
      cleaner.clean(false);
      closed = true;
    }
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
   * Get the device count.
   * @return returns the number of compute-capable devices
   * @throws CudaException on any error
   */
  public static native int getDeviceCount() throws CudaException;

  /**
   * Set the id of the current device.
   * <p>Note this is relative to CUDA_SET_VISIBLE_DEVICES, e.g. if
   * CUDA_SET_VISIBLE_DEVICES=1,0, and you call setDevice(0), you will get device 1.
   * <p>Note if RMM has been initialized and the requested device ID does not
   * match the device used to initialize RMM then this will throw an error.
   * @throws CudaException on any error
   */
  public static native void setDevice(int device) throws CudaException, CudfException;

  /**
   * Set the device for this thread to the appropriate one. Java loves threads, but cuda requires
   * each thread to have the device set explicitly or it falls back to CUDA_VISIBLE_DEVICES. Most
   * JNI calls through the cudf API will do this for you, but if you are writing your own JNI
   * calls that extend cudf you might want to call this before calling into your JNI APIs to
   * ensure that the device is set correctly.
   * @throws CudaException on any error
   */
  public static native void autoSetDevice() throws CudaException;

  /**
   * Calls cudaFree(0). This can be used to initialize the GPU after a setDevice()
   * @throws CudaException on any error
   */
  public static native void freeZero() throws CudaException;

  /**
   * Create a CUDA stream
   * @param isNonBlocking whether stream should be non-blocking with respect to the default stream
   * @return handle to a CUDA stream
   * @throws CudaException on any error
   */
  static native long createStream(boolean isNonBlocking) throws CudaException;

  /**
   * Destroy a CUDA stream
   * @param stream handle to the CUDA stream to destroy
   * @throws CudaException on any error
   */
  static native void destroyStream(long stream) throws CudaException;

  /**
   * Have this stream not execute new work until the work recorded in event completes.
   * @param stream the stream handle.
   * @param event the event handle.
   */
  static native void streamWaitEvent(long stream, long event) throws CudaException;

  /**
   * Block the thread until the pending execution on the stream completes
   * @param stream the stream handle
   * @throws CudaException on any error.
   */
  static native void streamSynchronize(long stream) throws CudaException;

  /**
   * Create a CUDA event
   * @param enableTiming true if timing should be enabled.
   * @param blockingSync true if blocking sync should be enabled.
   * @return handle to a CUDA event
   * @throws CudaException on any error
   */
  static native long createEvent(boolean enableTiming, boolean blockingSync) throws CudaException;

  /**
   * Destroy a CUDA event
   * @param event handle to the CUDA event to destroy
   * @throws CudaException on any error
   */
  static native void destroyEvent(long event) throws CudaException;

  /**
   * Check to see if the event happened or not.
   * @param event the event handle
   * @return true the event finished else false.
   * @throws CudaException on any error.
   */
  static native boolean eventQuery(long event) throws CudaException;

  /**
   * Reset the state of this event to be what is on the stream right now.
   * @param event the event handle
   * @param stream the stream handle
   * @throws CudaException on any error.
   */
  static native void eventRecord(long event, long stream) throws CudaException;

  /**
   * Block the thread until the execution recorded in the event is complete.
   * @param event the event handle
   * @throws CudaException on any error.
   */
  static native void eventSynchronize(long event) throws CudaException;

  /**
   * Copies bytes between buffers using the specified CUDA stream.
   * The copy has completed when this returns, but the memory copy could overlap with
   * operations occurring on other streams.
   * Specifying pointers that do not match the copy direction results in undefined behavior.
   * @param dst destination memory address
   * @param src source memory address
   * @param count size in bytes to copy
   * @param kind direction of transfer. {@link CudaMemcpyKind}
   * @param stream CUDA stream to use for the copy
   */
  static void memcpy(long dst, long src, long count, CudaMemcpyKind kind, Stream stream) {
    memcpyOnStream(dst, src, count, kind.getValue(), stream.getStream());
  }

  private static native void memcpyOnStream(long dst, long src, long count, int kind,
      long stream) throws CudaException;

  /**
   * Copies bytes between buffers using the specified CUDA stream.
   * The copy has not necessarily completed when this returns, but the memory copy could
   * overlap with operations occurring on other streams.
   * Specifying pointers that do not match the copy direction results in undefined behavior.
   * @param dst destination memory address
   * @param src source memory address
   * @param count size in bytes to copy
   * @param kind direction of transfer. {@link CudaMemcpyKind}
   * @param stream CUDA stream to use for the copy
   */
  static void asyncMemcpy(long dst, long src, long count, CudaMemcpyKind kind, Stream stream) {
    asyncMemcpyOnStream(dst, src, count, kind.getValue(), stream.getStream());
  }

  private static native void asyncMemcpyOnStream(long dst, long src, long count, int kind,
                                                 long stream) throws CudaException;

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
