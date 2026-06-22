/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;

/**
 * Base class that can be used to provide data dynamically to CUDF. This follows somewhat
 * closely with cudf::io::datasource. There are a few main differences.
 * <br/>
 * First this does not expose async device reads. It will call the non-async device read API
 * instead. This might be added in the future, but there was no direct use case for it in java
 * right now to warrant the added complexity.
 * <br/>
 * Second there is no implementation of the device read API that returns a buffer instead of
 * writing into one. This is not used by CUDF yet so testing an implementation that isn't used
 * didn't feel ideal. If it is needed we will add one in the future.
 */
public abstract class DataSource implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(DataSource.class);

  /**
   * This is used to keep track of the HostMemoryBuffers in java land so the C++ layer
   * does not have to do it.
   */
  private final HashMap<Long, HostMemoryBuffer> cachedBuffers = new HashMap<>();

  @Override
  public void close() {
    if (!cachedBuffers.isEmpty()) {
      throw new IllegalStateException("DataSource closed before all returned host buffers were closed");
    }
  }

  /**
   * Get the size of the source in bytes.
   */
  public abstract long size();

  /**
   * Read data from the source at the given offset. Return a HostMemoryBuffer for the data
   * that was read.
   * @param offset where to start reading from.
   * @param amount the maximum number of bytes to read.
   * @return a buffer that points to the data.
   * @throws IOException on any error.
   */
  public abstract HostMemoryBuffer hostRead(long offset, long amount) throws IOException;


  /**
   * Called when the buffer returned from hostRead is done. The default is to close the buffer.
   */
  protected void onHostBufferDone(HostMemoryBuffer buffer) {
    if (buffer != null) {
      buffer.close();
    }
  }

  /**
   * Read data from the source at the given offset into dest. Note that dest should not be closed,
   * and no reference to it can outlive the call to hostRead. The target amount to read is
   * dest's length.
   * @param offset the offset to start reading from in the source.
   * @param dest where to write the data.
   * @return the actual number of bytes written to dest.
   */
  public abstract long hostRead(long offset, HostMemoryBuffer dest) throws IOException;

  /**
   * Return true if this supports reading directly to the device else false. The default is
   * no device support. This cannot change dynamically. It is typically read just once.
   */
  public boolean supportsDeviceRead() {
    return false;
  }

  /**
   * Get the size cutoff between device reads and host reads when device reads are supported.
   * Anything larger than the cutoff will be a device read and anything smaller will be a
   * host read. By default, the cutoff is 0 so all reads will be device reads if device reads
   * are supported.
   */
  public long getDeviceReadCutoff() {
    return 0;
  }

  /**
   * Read data from the source at the given offset into dest. Note that dest should not be closed,
   * and no reference to it can outlive the call to hostRead. The target amount to read is
   * dest's length.
   * @param offset the offset to start reading from
   * @param dest where to write the data.
   * @param stream the stream to do the copy on.
   * @return the actual number of bytes written to dest.
   */
  public long deviceRead(long offset, DeviceMemoryBuffer dest,
                         Cuda.Stream stream) throws IOException {
    throw new IllegalStateException("Device read is not implemented");
  }

  /////////////////////////////////////////////////
  // Internal methods called from JNI
  /////////////////////////////////////////////////

  private static class NoopCleaner extends MemoryBuffer.MemoryBufferCleaner {
    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      return true;
    }

    @Override
    public boolean isClean() {
      return true;
    }
  }
  private static final NoopCleaner cleaner = new NoopCleaner();

  // Called from JNI
  private void onHostBufferDone(long bufferId) {
    HostMemoryBuffer hmb = cachedBuffers.remove(bufferId);
    if (hmb != null) {
      onHostBufferDone(hmb);
    } else {
      // Called from C++ destructor so avoid throwing...
      log.warn("Got a close callback for a buffer we could not find " + bufferId);
    }
  }

  // Called from JNI
  private long hostRead(long offset, long amount, long dst) throws IOException {
    if (amount < 0) {
      throw new IllegalArgumentException("Cannot allocate more than " + Long.MAX_VALUE + " bytes");
    }
    try (HostMemoryBuffer dstBuffer = new HostMemoryBuffer(dst, amount, cleaner)) {
      return hostRead(offset, dstBuffer);
    }
  }

  // Called from JNI
  private long[] hostReadBuff(long offset, long amount) throws IOException {
    if (amount < 0) {
      throw new IllegalArgumentException("Cannot read more than " + Long.MAX_VALUE + " bytes");
    }
    HostMemoryBuffer buff = hostRead(offset, amount);
    long[] ret = new long[3];
    if (buff != null) {
      long id = buff.id;
      if (cachedBuffers.put(id, buff) != null) {
        throw new IllegalStateException("Already had a buffer cached for " + buff);
      }
      ret[0] = buff.address;
      ret[1] = buff.length;
      ret[2] = id;
    } // else they are all 0 because java does that already
    return ret;
  }

  // Called from JNI
  private long deviceRead(long offset, long amount, long dst, long stream) throws IOException {
    if (amount < 0) {
      throw new IllegalArgumentException("Cannot read more than " + Long.MAX_VALUE + " bytes");
    }
    Cuda.Stream strm = Cuda.Stream.wrap(stream);
    try (DeviceMemoryBuffer dstBuffer = new DeviceMemoryBuffer(dst, amount, cleaner)) {
      return deviceRead(offset, dstBuffer, strm);
    }
  }
}
