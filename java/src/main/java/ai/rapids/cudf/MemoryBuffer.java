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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Abstract class for representing the Memory Buffer
 *
 * NOTE: MemoryBuffer is public to make it easier to work with the class hierarchy,
 * subclassing beyond what is included in CUDF is not recommended and not supported.
 */
abstract public class MemoryBuffer implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(MemoryBuffer.class);
  protected final long address;
  protected final long length;
  protected boolean closed = false;
  protected int refCount = 0;
  protected final MemoryBufferCleaner cleaner;
  protected final long id;

  public static abstract class MemoryBufferCleaner extends MemoryCleaner.Cleaner{}

  private static final class SlicedBufferCleaner extends MemoryBufferCleaner {
    private MemoryBuffer parent;

    SlicedBufferCleaner(MemoryBuffer parent) {
      this.parent = parent;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      if (parent != null) {
        if (logErrorIfNotClean) {
          log.error("A SLICED BUFFER WAS LEAKED(ID: " + id + " parent: " + parent + ")");
          logRefCountDebug("Leaked sliced buffer");
        }
        try {
          parent.close();
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          parent = null;
        }
        return true;
      }
      return false;
    }

    @Override
    public boolean isClean() {
      return parent == null;
    }
  }

  /**
   * This is a really ugly API, but it is possible that the lifecycle of a column of
   * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
   * tracking code that this is expected for this column, and big scary warnings should
   * not be printed when this happens.
   */
  public void noWarnLeakExpected() {
    if (cleaner != null) {
      cleaner.noWarnLeakExpected();
    }
  }

  /**
   * Constructor
   * @param address location in memory
   * @param length  size of this buffer
   * @param cleaner used to clean up the memory. May be null if no cleanup is needed.
   */
  protected MemoryBuffer(long address, long length, MemoryBufferCleaner cleaner) {
    this.address = address;
    this.length = length;
    this.cleaner = cleaner;
    if (cleaner != null) {
      this.id = cleaner.id;
      incRefCount();
      MemoryCleaner.register(this, cleaner);
    } else {
      this.id = -1;
    }
  }

  /**
   * Constructor
   * @param address location in memory
   * @param length  size of this buffer
   */
  protected MemoryBuffer(long address, long length) {
    this(address, length, (MemoryBufferCleaner)null);
  }

  /**
   * Internal constructor used when creating a slice.
   * @param address location in memory
   * @param length size of this buffer
   * @param parent the buffer that should be closed instead of closing this one.
   */
  protected MemoryBuffer(long address, long length, MemoryBuffer parent) {
    this(address, length, new SlicedBufferCleaner(parent));
  }

  /**
   * Returns the size of this buffer
   * @return - size
   */
  public final long getLength() {
    return length;
  }

  protected final void addressOutOfBoundsCheck(long address, long size, String type) {
    assert !closed : "Buffer is already closed " + Long.toHexString(this.address);
    assert size >= 0 : "A positive size is required";
    assert address >= this.address : "Start address is too low for " + type +
        " 0x" + Long.toHexString(address) + " < 0x" + Long.toHexString(this.address);
    assert (address + size) <= (this.address + length) : "End address is too high for " + type +
        " 0x" + Long.toHexString(address + size) + " < 0x" + Long.toHexString(this.address + length);
  }

  /**
   * Returns the location of the data pointed to by this buffer
   * @return - data address
   */
  public final long getAddress() {
    return address;
  }

  /**
   * Slice off a part of the buffer. Note that this is a zero copy operation and all
   * slices must be closed along with the original buffer before the memory is released.
   * So use this with some caution.
   *
   * Note that [[DeviceMemoryBuffer]] and [[HostMemoryBuffer]] support slicing, and override this
   * function.
   *
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a slice of the original buffer that will need to be closed independently
   */
  public abstract MemoryBuffer slice(long offset, long len);

  /**
   * Close this buffer and free memory
   */
  public synchronized void close() {
    if (cleaner != null) {
      refCount--;
      cleaner.delRef();
      if (refCount == 0) {
        cleaner.clean(false);
        closed = true;
      } else if (refCount < 0) {
        cleaner.logRefCountDebug("double free " + this);
        throw new IllegalStateException("Close called too many times " + this);
      }
    }
  }

  @Override
  public String toString() {
    long id = -1;
    if (cleaner != null) {
      id = cleaner.id;
    }
    String name = this.getClass().getSimpleName();
    return name + "{" +
        "address=0x" + Long.toHexString(address) +
        ", length=" + length +
        ", id=" + id + "}";
  }

  /**
   * Increment the reference count for this column.  You need to call close on this
   * to decrement the reference count again.
   */
  public synchronized void incRefCount() {
    refCount++;
    cleaner.addRef();
  }

  // visible for testing
  synchronized int getRefCount() {
    return refCount;
  }
}
