/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * This class does bit manipulation using byte arithmetic
 */
final class BitVectorHelper {

  /**
   * Shifts that to the left by the required bits then appends to this
   */
  static void append(HostMemoryBuffer src, HostMemoryBuffer dst, long dstOffset, long rows) {
    assert dst.length * 8 - dstOffset >= rows : "validity vector bigger then available space on " +
        "dst: " + (dst.length * 8 - dstOffset) + " copying space needed: " + rows;
    long dstByteIndex = dstOffset / 8;
    int shiftBits = (int) (dstOffset % 8);
    if (shiftBits > 0) {
      shiftSrcLeftAndWriteToDst(src, dst, dstByteIndex, shiftBits, rows);
    } else {
      dst.copyFromHostBuffer(dstByteIndex, src, 0, getValidityLengthInBytes(rows));
    }
  }

  /**
   * Shifts the src to the left by the given bits and writes 'length' bytes to the destination
   */
  private static void shiftSrcLeftAndWriteToDst(HostMemoryBuffer src, HostMemoryBuffer dst,
                                                long dstOffset, int shiftByBits, long length) {
    assert shiftByBits > 0 && shiftByBits < 8 : "shiftByBits out of range";
    int dstMask = 0xFF >> (8 - shiftByBits);
    // the mask to save the left side of the bits before we shift
    int srcLeftMask = dstMask << (8 - shiftByBits);
    int valueFromTheLeftOfTheLastByte = dst.getByte(dstOffset) & dstMask;
    long i;
    long byteLength = getValidityLengthInBytes(length);
    for (i = 0; i < byteLength; i++) {
      int b = src.getByte(i);
      int fallingBitsOnTheLeft = b & srcLeftMask;
      b <<= shiftByBits;
      b |= valueFromTheLeftOfTheLastByte;
      dst.setByte(dstOffset + i, (byte) b);
      valueFromTheLeftOfTheLastByte = fallingBitsOnTheLeft >>> (8 - shiftByBits);
    }
    if (((length % 8) + shiftByBits > 8) || length % 8 == 0) {
      /*
          Only if the last byte has data that has been shifted to spill over to the next
          byte execute the
          following statement.
       */
      dst.setByte(dstOffset + i, (byte) (valueFromTheLeftOfTheLastByte | ~dstMask));
    }
  }

  /**
   * This method returns the length in bytes needed to represent X number of rows
   * e.g. getValidityLengthInBytes(5) => 1 byte
   * getValidityLengthInBytes(7) => 1 byte
   * getValidityLengthInBytes(14) => 2 bytes
   */
  static long getValidityLengthInBytes(long rows) {
    return (rows + 7) / 8;
  }

  /**
   * This method returns the allocation size of the validity vector which is 64-byte aligned
   * e.g. getValidityAllocationSizeInBytes(5) => 64 bytes
   * getValidityAllocationSizeInBytes(14) => 64 bytes
   * getValidityAllocationSizeInBytes(65) => 128 bytes
   */
  static long getValidityAllocationSizeInBytes(long rows) {
    long numBytes = getValidityLengthInBytes(rows);
    return ((numBytes + 63) / 64) * 64;
  }

  /**
   * Set the validity bit to null for the given index.
   * @param valid the buffer to set it in.
   * @param index the index to set it at.
   * @return 1 if validity changed else 0 if it already was null.
   */
  static int setNullAt(HostMemoryBuffer valid, long index) {
    long bucket = index / 8;
    byte currentByte = valid.getByte(bucket);
    int bitmask = ~(1 << (index % 8));
    int ret = (currentByte >> index) & 0x1;
    currentByte &= bitmask;
    valid.setByte(bucket, currentByte);
    return ret;
  }

  static boolean isNull(HostMemoryBuffer valid, long index) {
    int b = valid.getByte(index / 8);
    int i = b & (1 << (index % 8));
    return i == 0;
  }
}
