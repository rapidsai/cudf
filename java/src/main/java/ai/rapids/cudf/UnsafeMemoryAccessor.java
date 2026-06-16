/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Field;

/**
 * UnsafeMemory Accessor for accessing memory on host
 */
class UnsafeMemoryAccessor {

  public static final long BYTE_ARRAY_OFFSET;
  public static final long SHORT_ARRAY_OFFSET;
  public static final long INT_ARRAY_OFFSET;
  public static final long LONG_ARRAY_OFFSET;
  public static final long FLOAT_ARRAY_OFFSET;
  public static final long DOUBLE_ARRAY_OFFSET;
  private static final sun.misc.Unsafe UNSAFE;
  /**
   * Limits the number of bytes to copy per {@link sun.misc.Unsafe#copyMemory(long, long, long)} to
   * allow safepoint polling during a large copy.
   */
  private static final long UNSAFE_COPY_THRESHOLD = 1024L * 1024L;
  private static Logger log = LoggerFactory.getLogger(UnsafeMemoryAccessor.class);

  static {
    sun.misc.Unsafe unsafe = null;
    try {
      Field unsafeField = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
      unsafeField.setAccessible(true);
      unsafe = (sun.misc.Unsafe) unsafeField.get(null);
      BYTE_ARRAY_OFFSET = unsafe.arrayBaseOffset(byte[].class);
      SHORT_ARRAY_OFFSET = unsafe.arrayBaseOffset(short[].class);
      INT_ARRAY_OFFSET = unsafe.arrayBaseOffset(int[].class);
      LONG_ARRAY_OFFSET = unsafe.arrayBaseOffset(long[].class);
      FLOAT_ARRAY_OFFSET = unsafe.arrayBaseOffset(float[].class);
      DOUBLE_ARRAY_OFFSET = unsafe.arrayBaseOffset(double[].class);
    } catch (Throwable t) {
      log.error("Failed to get unsafe object, got this error: ", t);
      UNSAFE = null;
      throw new NullPointerException("Failed to get unsafe object, got this error: " + t.getMessage());
    }
    UNSAFE = unsafe;
  }

  /**
   * Get the system memory page size.
   * @return system memory page size in bytes
   */
  public static int pageSize() {
    return UNSAFE.pageSize();
  }

  /**
   * Allocate bytes on host
   * @param bytes - number of bytes to allocate
   * @return - allocated address
   */
  public static long allocate(long bytes) {
    return UNSAFE.allocateMemory(bytes);
  }

  /**
   * Free memory at that location
   * @param address - memory location
   */
  public static void free(long address) {
    UNSAFE.freeMemory(address);
  }

  /**
   * Sets the values at this address repeatedly
   * @param address - memory location
   * @param size    - number of bytes to set
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static void setMemory(long address, long size, byte value) {
    UNSAFE.setMemory(address, size, value);
  }

  /**
   * Sets the Byte value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static void setByte(long address, byte value) {
    UNSAFE.putByte(address, value);
  }

  /**
   * Sets an array of bytes.
   * @param address - memory address
   * @param values  to be set
   * @param offset  index into values to start at.
   * @param len     the number of bytes to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setBytes(long address, byte[] values, long offset, long len) {
    copyMemory(values, UnsafeMemoryAccessor.BYTE_ARRAY_OFFSET + offset,
        null, address, len);
  }

  /**
   * Returns the Byte value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static byte getByte(long address) {
    return UNSAFE.getByte(address);
  }

  /**
   * Copy out an array of bytes.
   * @param dst       where to write the data
   * @param dstOffset index into values to start writing at.
   * @param address   src memory address
   * @param len       the number of bytes to copy
   * @throws IndexOutOfBoundsException
   */
  public static void getBytes(byte[] dst, long dstOffset, long address, long len) {
    copyMemory(null, address,
        dst, UnsafeMemoryAccessor.BYTE_ARRAY_OFFSET + dstOffset, len);
  }

  /**
   * Returns the Integer value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static int getInt(long address) {
    return UNSAFE.getInt(address);
  }

  /**
   * Copy out an array of ints.
   * @param dst       where to write the data
   * @param dstIndex  index into values to start writing at.
   * @param address   src memory address
   * @param count     the number of ints to copy
   * @throws IndexOutOfBoundsException
   */
  public static void getInts(int[] dst, long dstIndex, long address, int count) {
    copyMemory(null, address,
            dst, UnsafeMemoryAccessor.INT_ARRAY_OFFSET + (dstIndex * 4), count * 4);
  }

  /**
   * Sets the Integer value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static void setInt(long address, int value) {
    UNSAFE.putInt(address, value);
  }

  /**
   * Sets an array of ints.
   * @param address memory address
   * @param values  to be set
   * @param offset  index into values to start at.
   * @param len     the number of ints to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setInts(long address, int[] values, long offset, long len) {
    copyMemory(values, UnsafeMemoryAccessor.INT_ARRAY_OFFSET + (offset * 4),
        null, address, len * 4);
  }

  /**
   * Sets the Long value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static void setLong(long address, long value) {
    UNSAFE.putLong(address, value);
  }

  /**
   * Sets an array of longs.
   * @param address memory address
   * @param values  to be set
   * @param offset  index into values to start at
   * @param len     the number of longs to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setLongs(long address, long[] values, long offset, long len) {
    copyMemory(values, UnsafeMemoryAccessor.LONG_ARRAY_OFFSET + (offset * 8),
        null, address, len * 8);
  }

  /**
   * Returns the Long value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static long getLong(long address) {
    return UNSAFE.getLong(address);
  }

  /**
   * Copy out an array of longs.
   * @param dst       where to write the data
   * @param dstIndex  index into values to start writing at.
   * @param address   src memory address
   * @param count     the number of longs to copy
   * @throws IndexOutOfBoundsException
   */
  public static void getLongs(long[] dst, long dstIndex, long address, int count) {
    copyMemory(null, address,
        dst, UnsafeMemoryAccessor.LONG_ARRAY_OFFSET + (dstIndex * 8), count * 8);
  }

  /**
   * Returns the Short value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static short getShort(long address) {
    return UNSAFE.getShort(address);
  }

  /**
   * Sets the Short value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static void setShort(long address, short value) {
    UNSAFE.putShort(address, value);
  }

  /**
   * Sets an array of shorts.
   * @param address memory address
   * @param values  to be set
   * @param offset  index into values to start at
   * @param len     the number of shorts to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setShorts(long address, short[] values, long offset, long len) {
    copyMemory(values, UnsafeMemoryAccessor.SHORT_ARRAY_OFFSET + (offset * 2),
        null, address, len * 2);
  }

  /**
   * Sets the Double value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static void setDouble(long address, double value) {
    UNSAFE.putDouble(address, value);
  }

  /**
   * Sets an array of doubles.
   * @param address memory address
   * @param values  to be set
   * @param offset  index into values to start at
   * @param len     the number of doubles to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setDoubles(long address, double[] values, long offset, long len) {
    copyMemory(values, UnsafeMemoryAccessor.DOUBLE_ARRAY_OFFSET + (offset * 8),
        null, address, len * 8);
  }

  /**
   * Returns the Double value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static double getDouble(long address) {
    return UNSAFE.getDouble(address);
  }

  /**
   * Returns the Float value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static float getFloat(long address) {
    return UNSAFE.getFloat(address);
  }

  /**
   * Sets the Float value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static void setFloat(long address, float value) {
    UNSAFE.putFloat(address, value);
  }

  /**
   * Sets an array of floats.
   * @param address memory address
   * @param values  to be set
   * @param offset  the index in values to start at
   * @param len     the number of floats to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setFloats(long address, float[] values, long offset, long len) {
    copyMemory(values, UnsafeMemoryAccessor.FLOAT_ARRAY_OFFSET + (offset * 4),
        null, address, len * 4);
  }

  /**
   * Returns the Boolean value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static boolean getBoolean(long address) {
    return getByte(address) != 0 ? true : false;
  }

  /**
   * Sets the Boolean value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static void setBoolean(long address, boolean value) {
    setByte(address, (byte) (value ? 1 : 0));
  }


  /**
   * Copy memory from one address to the other.
   */
  public static void copyMemory(Object src, long srcOffset, Object dst, long dstOffset,
                                long length) {
    // Check if dstOffset is before or after srcOffset to determine if we should copy
    // forward or backwards. This is necessary in case src and dst overlap.
    if (dstOffset < srcOffset) {
      while (length > 0) {
        long size = Math.min(length, UNSAFE_COPY_THRESHOLD);
        UNSAFE.copyMemory(src, srcOffset, dst, dstOffset, size);
        length -= size;
        srcOffset += size;
        dstOffset += size;
      }
    } else {
      srcOffset += length;
      dstOffset += length;
      while (length > 0) {
        long size = Math.min(length, UNSAFE_COPY_THRESHOLD);
        srcOffset -= size;
        dstOffset -= size;
        UNSAFE.copyMemory(src, srcOffset, dst, dstOffset, size);
        length -= size;
      }

    }
  }
}
