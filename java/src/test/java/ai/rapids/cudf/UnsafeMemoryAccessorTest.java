/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

@Tag("noSanitizer")
public class UnsafeMemoryAccessorTest {
  @Test
  public void testAllocate() {
    long address = UnsafeMemoryAccessor.allocate(3);
    try {
      assertNotEquals(0, address);
    } finally {
      UnsafeMemoryAccessor.free(address);
    }
  }

  @Test
  public void setByteAndGetByte() {
    long address = UnsafeMemoryAccessor.allocate(2);
    try {
      UnsafeMemoryAccessor.setByte(address, (byte) 34);
      UnsafeMemoryAccessor.setByte(address + 1, (byte) 63);
      Byte b = UnsafeMemoryAccessor.getByte(address);
      assertEquals((byte) 34, b);
      b = UnsafeMemoryAccessor.getByte(address + 1);
      assertEquals((byte) 63, b);
    } finally {
      UnsafeMemoryAccessor.free(address);
    }
  }

  @Test
  public void setIntAndGetInt() {
    long address = UnsafeMemoryAccessor.allocate(2 * 4);
    try {
      UnsafeMemoryAccessor.setInt(address, 2);
      UnsafeMemoryAccessor.setInt(address + 4, 4);
      int v = UnsafeMemoryAccessor.getInt(address);
      assertEquals(2, v);
      v = UnsafeMemoryAccessor.getInt(address + 4);
      assertEquals(4, v);
    } finally {
      UnsafeMemoryAccessor.free(address);
    }
  }

  @Test
  public void setAndGetInts() {
    int numInts = 289;
    long address = UnsafeMemoryAccessor.allocate(numInts * 4);
    try {
      for (int i = 0; i < numInts; i++) {
        UnsafeMemoryAccessor.setInt(address + i * 4, i);
      }
      int[] ints = new int[numInts];
      UnsafeMemoryAccessor.getInts(ints, 0, address, numInts);
      for (int i = 0; i < numInts; i++) {
        assertEquals(i, ints[i]);
      }
    } finally {
      UnsafeMemoryAccessor.free(address);
    }
  }

  @Test
  public void setMemoryValue() {
    long address = UnsafeMemoryAccessor.allocate(4);
    try {
      UnsafeMemoryAccessor.setMemory(address, 4, (byte) 1);
      int v = UnsafeMemoryAccessor.getInt(address);
      assertEquals(16843009, v);
    } finally {
      UnsafeMemoryAccessor.free(address);
    }
  }

  @Test
  public void testGetLongs() {
    int numLongs = 257;
    long address = UnsafeMemoryAccessor.allocate(numLongs * 8);
    for (int i = 0; i < numLongs; ++i) {
      UnsafeMemoryAccessor.setLong(address + (i * 8), i);
    }
    long[] result = new long[numLongs];
    UnsafeMemoryAccessor.getLongs(result, 0, address, numLongs);
    for (int i = 0; i < numLongs; ++i) {
      assertEquals(i, result[i]);
    }
    UnsafeMemoryAccessor.getLongs(result, 1,
        address + ((numLongs - 1) * 8), 1);
    for (int i = 0; i < numLongs; ++i) {
      long expected = (i == 1) ? numLongs - 1 : i;
      assertEquals(expected, result[i]);
    }
  }
}
