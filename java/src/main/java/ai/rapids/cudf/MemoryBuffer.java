/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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
 */
abstract class MemoryBuffer implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(MemoryBuffer.class);
  protected final long address;
  protected final long length;
  protected boolean closed = false;
  protected int refCount = 0;
  protected final MemoryCleaner.Cleaner cleaner;

  private static final class SlicedBufferCleaner extends MemoryCleaner.Cleaner {
    private MemoryBuffer parent;

    SlicedBufferCleaner(MemoryBuffer parent) {
      this.parent = parent;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (parent != null) {
        parent.close();
        parent = null;
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("WE LEAKED A SLICED BUFFER!!!!");
        logRefCountDebug("Leaked sliced buffer");
      }
      return neededCleanup;
    }
  }

  /**
   * This is a really ugly API, but it is possible that the lifecycle of a column of
   * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
   * tracking code that this is expected for this column, and big scary warnings should
   * not be printed when this happens.
   */
  public void noWarnLeakExpected() {
    cleaner.noWarnLeakExpected();
  }

  /**
   * Public constructor
   * @param address location in memory
   * @param length  size of this buffer
   */
  protected MemoryBuffer(long address, long length, MemoryCleaner.Cleaner cleaner) {
    this.address = address;
    this.length = length;
    this.cleaner = cleaner;
    refCount++;
    cleaner.addRef();
    MemoryCleaner.register(this, cleaner);
  }

  /**
   * Internal constructor used when creating a slice.
   * @param address location in memory
   * @param length size of this buffer
   * @param parent the buffer that should be closed instead of closing this one.
   */
  protected MemoryBuffer(long address, long length, MemoryBuffer parent) {
    this.address = address;
    this.length = length;
    this.cleaner = new SlicedBufferCleaner(parent);
    refCount++;
    cleaner.addRef();
    MemoryCleaner.register(this, cleaner);
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
    assert address >= this.address : "Start address is too low for " + type +
        " 0x" + Long.toHexString(address) + " < 0x" + Long.toHexString(this.address);
    assert (address + size) <= (this.address + length) : "End address is too high for " + type +
        " 0x" + Long.toHexString(address + size) + " < 0x" + Long.toHexString(this.address + length);
  }

  /**
   * Returns the location of the data pointed to by this buffer
   * @return - data address
   */
  final long getAddress() {
    return address;
  }

  /**
   * Close this buffer and free memory
   */
  public final void close() {
    refCount--;
    cleaner.delRef();
    if (refCount == 0) {
      cleaner.clean(false);
      closed = true;
    } else if (refCount < 0) {
      log.error("Close called too many times on {}", this);
      cleaner.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times");
    }
  }

  @Override
  public String toString() {
    return "MemoryBuffer{" +
        "address=0x" + Long.toHexString(address) +
        ", length=" + length +
        '}';
  }
}
