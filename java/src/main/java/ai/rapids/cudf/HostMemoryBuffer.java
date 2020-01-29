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

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * This class holds an off-heap buffer in the host/CPU memory.
 * Please note that instances must be explicitly closed or native memory will be leaked!
 *
 * Internally this class may optionally use PinnedMemoryPool to allocate and free the memory
 * it uses. To try to use the pinned memory pool for allocations set the java system property
 * ai.rapids.cudf.prefer-pinned to true.
 *
 * Be aware that the off heap memory limits set by java do nto apply to these buffers.
 */
public class HostMemoryBuffer extends MemoryBuffer {
  private static final boolean defaultPreferPinned = Boolean.getBoolean(
      "ai.rapids.cudf.prefer-pinned");
  private static final Logger log = LoggerFactory.getLogger(HostMemoryBuffer.class);

  // Make sure we loaded the native dependencies so we have a way to create a ByteBuffer
  {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * This will turn an address into a ByteBuffer.  The buffer will NOT own the memory
   * so closing it has no impact on the underlying memory. It should never
   * be used if the corresponding HostMemoryBuffer is closed.
   */
  private static native ByteBuffer wrapRangeInBuffer(long address, long len);

  private static final class HostBufferCleaner extends MemoryBufferCleaner {
    private long address;
    private final long length;

    HostBufferCleaner(long address, long length) {
      this.address = address;
      this.length = length;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (address != 0) {
        UnsafeMemoryAccessor.free(address);
        MemoryListener.hostDeallocation(length, getId());
        address = 0;
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A HOST BUFFER WAS LEAKED!!!!");
        logRefCountDebug("Leaked host buffer");
      }
      return neededCleanup;
    }
  }

  /**
   * Allocate memory, but be sure to close the returned buffer to avoid memory leaks.
   * @param bytes size in bytes to allocate
   * @param preferPinned If set to true, the pinned memory pool will be used if possible with a
   *                    fallback to off-heap memory.  If set to false, the allocation will always
   *                    be from off-heap memory.
   * @return the newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes, boolean preferPinned) {
    if (preferPinned) {
      HostMemoryBuffer pinnedBuffer = PinnedMemoryPool.tryAllocate(bytes);
      if (pinnedBuffer != null) {
        return pinnedBuffer;
      }
    }
    return new HostMemoryBuffer(UnsafeMemoryAccessor.allocate(bytes), bytes);
  }

  /**
   * Allocate memory, but be sure to close the returned buffer to avoid memory leaks. Pinned memory
   * will be preferred for allocations if the java system property ai.rapids.cudf.prefer-pinned is
   * set to true.
   * @param bytes size in bytes to allocate
   * @return the newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes) {
    return allocate(bytes, defaultPreferPinned);
  }

  HostMemoryBuffer(long address, long length) {
    this(address, length, new HostBufferCleaner(address, length));
  }

  HostMemoryBuffer(long address, long length, MemoryBufferCleaner cleaner) {
    super(address, length, cleaner);
    if (length > 0) {
      MemoryListener.hostAllocation(length, id);
    }
  }

  private HostMemoryBuffer(long address, long lengthInBytes, HostMemoryBuffer parent) {
    super(address, lengthInBytes, parent);
    // This is a slice so we are not going to mark it as allocated
  }

  /**
   * Return a ByteBuffer that provides access to the underlying memory.  Please note: if the buffer
   * is larger than a ByteBuffer can handle (2GB) an exception will be thrown.  Also
   * be aware that the ByteBuffer will be in native endian order, which is different from regular
   * ByteBuffers that are big endian by default.
   */
  public final ByteBuffer asByteBuffer() {
    assert length <= Integer.MAX_VALUE : "2GB limit on ByteBuffers";
    return asByteBuffer(0, (int) length);
  }

  /**
   * Return a ByteBuffer that provides access to the underlying memory.  Be aware that the
   * ByteBuffer will be in native endian order, which is different from regular
   * ByteBuffers that are big endian by default.
   * @param offset the offset to start at
   * @param length how many bytes to include.
   */
  public final ByteBuffer asByteBuffer(long offset, int length) {
    addressOutOfBoundsCheck(address + offset, length, "asByteBuffer");
    return wrapRangeInBuffer(address + offset, length)
        .order(ByteOrder.nativeOrder());
  }

  /**
   * Copy the contents of the given buffer to this buffer
   * @param destOffset offset in bytes in this buffer to start copying to
   * @param srcData    Buffer to be copied from
   * @param srcOffset  offset in bytes to start copying from in srcData
   * @param length     number of bytes to copy
   */
  public final void copyFromHostBuffer(long destOffset, HostMemoryBuffer srcData, long srcOffset,
      long length) {
    addressOutOfBoundsCheck(address + destOffset, length, "copy from dest");
    srcData.addressOutOfBoundsCheck(srcData.address + srcOffset, length, "copy from source");
    UnsafeMemoryAccessor.copyMemory(null, srcData.address + srcOffset, null,
        address + destOffset, length);
  }

  /**
   * Copy len bytes from in to this buffer.
   * @param destOffset  offset in bytes in this buffer to start copying to
   * @param in input stream to copy bytes from
   * @param byteLength number of bytes to copy
   */
  final void copyFromStream(long destOffset, InputStream in, long byteLength) throws IOException {
    addressOutOfBoundsCheck(address + destOffset, byteLength, "copy from stream");
    byte[] arrayBuffer = new byte[(int) Math.min(1024 * 128, byteLength)];
    long left = byteLength;
    while (left > 0) {
      int amountToCopy = (int) Math.min(arrayBuffer.length, left);
      int amountRead = in.read(arrayBuffer, 0, amountToCopy);
      if (amountRead < 0) {
        throw new EOFException();
      }
      setBytes(destOffset, arrayBuffer, 0, amountRead);
      destOffset += amountRead;
      left -= amountRead;
    }
  }

  /**
   * Returns the byte value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final byte getByte(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "getByte");
    return UnsafeMemoryAccessor.getByte(requestedAddress);
  }

  /**
   * Sets the byte value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setByte(long offset, byte value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "setByte");
    UnsafeMemoryAccessor.setByte(requestedAddress, value);
  }

  /**
   * Copy a set of bytes to an array from the buffer starting at offset.
   * @param dstOffset the offset from the address to start copying to
   * @param dst       the data to be copied.
   */
  public final void getBytes(byte[] dst, long dstOffset, long srcOffset, long len) {
    assert len >= 0;
    assert len <= dst.length - dstOffset;
    assert srcOffset >= 0;
    long requestedAddress = this.address + srcOffset;
    addressOutOfBoundsCheck(requestedAddress, len, "getBytes");
    UnsafeMemoryAccessor.getBytes(dst, dstOffset, requestedAddress, len);
  }

  /**
   * Copy a set of bytes from an array into the buffer at offset.
   * @param offset the offset from the address to start copying to
   * @param data   the data to be copied.
   */
  public final void setBytes(long offset, byte[] data, long srcOffset, long len) {
    assert len >= 0 : "length is not allowed " + len;
    assert len <= data.length - srcOffset;
    assert srcOffset >= 0;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len, "setBytes");
    UnsafeMemoryAccessor.setBytes(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Short value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  final short getShort(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 2, "getShort");
    return UnsafeMemoryAccessor.getShort(requestedAddress);
  }

  /**
   * Sets the Short value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  final void setShort(long offset, short value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 2, "setShort");
    UnsafeMemoryAccessor.setShort(requestedAddress, value);
  }

  /**
   * Copy a set of shorts from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index in data to start at.
   */
  final void setShorts(long offset, short[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 2, "setShorts");
    UnsafeMemoryAccessor.setShorts(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Integer value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  final int getInt(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "getInt");
    return UnsafeMemoryAccessor.getInt(requestedAddress);
  }

  /**
   * Sets the Integer value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  final void setInt(long offset, int value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "setInt");
    UnsafeMemoryAccessor.setInt(requestedAddress, value);
  }

  /**
   * Copy a set of ints from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at
   */
  final void setInts(long offset, int[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 4, "setInts");
    UnsafeMemoryAccessor.setInts(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Long value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  final long getLong(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "setLong");
    return UnsafeMemoryAccessor.getLong(requestedAddress);
  }

  /**
   * Sets the Long value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  final void setLong(long offset, long value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "getLong");
    UnsafeMemoryAccessor.setLong(requestedAddress, value);
  }

  /**
   * Copy a set of longs from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at.
   */
  final void setLongs(long offset, long[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 8, "setLongs");
    UnsafeMemoryAccessor.setLongs(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Float value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  final float getFloat(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "getFloat");
    return UnsafeMemoryAccessor.getFloat(requestedAddress);
  }

  /**
   * Sets the Float value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  final void setFloat(long offset, float value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "setFloat");
    UnsafeMemoryAccessor.setFloat(requestedAddress, value);
  }

  /**
   * Copy a set of floats from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at
   */
  final void setFloats(long offset, float[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 4, "setFloats");
    UnsafeMemoryAccessor.setFloats(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Double value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  final double getDouble(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "getDouble");
    return UnsafeMemoryAccessor.getDouble(requestedAddress);
  }

  /**
   * Sets the Double value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  final void setDouble(long offset, double value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "setDouble");
    UnsafeMemoryAccessor.setDouble(requestedAddress, value);
  }

  /**
   * Copy a set of doubles from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at
   */
  final void setDoubles(long offset, double[] data, long srcOffset, long len) {
    assert len > 0;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 8, "setDoubles");
    UnsafeMemoryAccessor.setDoubles(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Boolean value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  final boolean getBoolean(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "getBoolean");
    return UnsafeMemoryAccessor.getBoolean(requestedAddress);
  }

  /**
   * Sets the Boolean value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  final void setBoolean(long offset, boolean value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "setBoolean");
    UnsafeMemoryAccessor.setBoolean(requestedAddress, value);
  }

  /**
   * Sets the values in this buffer repeatedly
   * @param offset - offset from the address
   * @param length - number of bytes to set
   * @param value  - value to be set
   */
  final void setMemory(long offset, long length, byte value) {
    addressOutOfBoundsCheck(address + offset, length, "set memory");
    UnsafeMemoryAccessor.setMemory(address + offset, length, value);
  }

  final void copyFromMemory(long fromAddress, long len) {
    addressOutOfBoundsCheck(address, len, "copy from memory");
    UnsafeMemoryAccessor.copyMemory(null, fromAddress, null, address, len);
  }

  /**
   * Method to copy from a DeviceMemoryBuffer to a HostMemoryBuffer
   * @param deviceMemoryBuffer - Buffer to copy data from
   */
  final void copyFromDeviceBuffer(BaseDeviceMemoryBuffer deviceMemoryBuffer) {
    addressOutOfBoundsCheck(address, deviceMemoryBuffer.length, "copy range dest");
    assert !deviceMemoryBuffer.closed;
    Cuda.memcpy(address, deviceMemoryBuffer.address, deviceMemoryBuffer.length,
        CudaMemcpyKind.DEVICE_TO_HOST);
  }

  /**
   * Slice off a part of the host buffer.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a host buffer that will need to be closed independently from this buffer.
   */
  final HostMemoryBuffer slice(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");
    refCount++;
    cleaner.addRef();
    return new HostMemoryBuffer(address + offset, len, this);
  }

  /**
   * Slice off a part of the host buffer, actually making a copy of the data.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a host buffer that will need to be closed independently from this buffer.
   */
  final HostMemoryBuffer sliceWithCopy(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");

    HostMemoryBuffer ret = null;
    boolean success = false;
    try {
      ret = allocate(len);
      UnsafeMemoryAccessor.copyMemory(null, address + offset, null, ret.getAddress(), len);
      success = true;
      return ret;
    } finally {
      if (!success && ret != null) {
        ret.close();
      }
    }
  }
}
