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

/**
 * Abstract class for representing the Memory Buffer
 */
abstract class MemoryBuffer implements AutoCloseable {
  protected final long address;
  protected final long length;
  protected boolean closed = false;

  /**
   * Public constructor
   * @param address - location in memory
   * @param length  - size of this buffer
   */
  protected MemoryBuffer(long address, long length) {
    this.address = address;
    this.length = length;
  }

  /**
   * Returns the size of this buffer
   * @return - size
   */
  public long getLength() {
    return length;
  }

  protected final void addressOutOfBoundsCheck(long address, long size, String type) {
    assert !closed;
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
   * Actually close the buffer.
   */
  protected abstract void doClose();

  /**
   * Close this buffer and free memory
   */
  public final void close() {
    if (!closed) {
      doClose();
      closed = true;
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
