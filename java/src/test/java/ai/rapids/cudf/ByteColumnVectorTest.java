/*
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

import ai.rapids.cudf.HostColumnVector.Builder;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ByteColumnVectorTest extends CudfTestBase {

  @Test
  public void testCreateColumnVectorBuilder() {
    try (HostColumnVector shortColumnVector = HostColumnVector.build(DType.INT8, 3,
        (b) -> b.append((byte) 1))) {
      assertFalse(shortColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    try (HostColumnVector byteColumnVector = HostColumnVector.fromBytes(new byte[]{2, 3, 5})) {
      assertFalse(byteColumnVector.hasNulls());
      assertEquals(byteColumnVector.getByte(0), 2);
      assertEquals(byteColumnVector.getByte(1), 3);
      assertEquals(byteColumnVector.getByte(2), 5);
    }
  }

  @Test
  public void testUnsignedArrayAllocation() {
    try (HostColumnVector v = HostColumnVector.fromUnsignedBytes(new byte[]{(byte)0xff, (byte)128, 5})) {
      assertFalse(v.hasNulls());
      assertEquals(0xff, Byte.toUnsignedInt(v.getByte(0)), 0xff);
      assertEquals(128, Byte.toUnsignedInt(v.getByte(1)), 128);
      assertEquals(5, Byte.toUnsignedInt(v.getByte(2)), 5);
    }
  }

  @Test
  public void testAppendRepeatingValues() {
    try (HostColumnVector byteColumnVector = HostColumnVector.build(DType.INT8, 3,
        (b) -> b.append((byte) 2, 3L))) {
      assertFalse(byteColumnVector.hasNulls());
      assertEquals(byteColumnVector.getByte(0), 2);
      assertEquals(byteColumnVector.getByte(1), 2);
      assertEquals(byteColumnVector.getByte(2), 2);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    try (HostColumnVector byteColumnVector = HostColumnVector.fromBytes(new byte[]{2, 3, 5})) {
      assertThrows(AssertionError.class, () -> byteColumnVector.getByte(3));
      assertFalse(byteColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    try (HostColumnVector byteColumnVector = HostColumnVector.fromBytes(new byte[]{2, 3, 5})) {
      assertFalse(byteColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> byteColumnVector.getByte(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    try (HostColumnVector byteColumnVector = HostColumnVector.fromBoxedBytes(
        new Byte[]{2, 3, 4, 5, 6, 7, null, null})) {
      assertTrue(byteColumnVector.hasNulls());
      assertEquals(2, byteColumnVector.getNullCount());
      for (int i = 0; i < 6; i++) {
        assertFalse(byteColumnVector.isNull(i));
      }
      assertTrue(byteColumnVector.isNull(6));
      assertTrue(byteColumnVector.isNull(7));
    }
  }

  @Test
  public void testAddingUnsignedNullValues() {
    try (HostColumnVector byteColumnVector = HostColumnVector.fromBoxedUnsignedBytes(
        new Byte[]{2, 3, 4, 5, (byte)128, (byte)254, null, null})) {
      assertTrue(byteColumnVector.hasNulls());
      assertEquals(2, byteColumnVector.getNullCount());
      for (int i = 0; i < 6; i++) {
        assertFalse(byteColumnVector.isNull(i));
      }
      assertEquals(128, Byte.toUnsignedInt(byteColumnVector.getByte(4)));
      assertEquals(254, Byte.toUnsignedInt(byteColumnVector.getByte(5)));
      assertTrue(byteColumnVector.isNull(6));
      assertTrue(byteColumnVector.isNull(7));
    }
  }

  @Test
  public void testCastToByte() {
    final int[] DATES = {17897}; //Jan 01, 2019

    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(4.3, 3.8, 8);
         ColumnVector shortColumnVector = ColumnVector.fromShorts(new short[]{100});
         ColumnVector dateColumnVector = ColumnVector.daysFromInts(DATES);
         ColumnVector byteColumnVector1 = doubleColumnVector.asBytes();
         ColumnVector byteColumnVector2 = shortColumnVector.asBytes();
         ColumnVector byteColumnVector3 = dateColumnVector.asBytes();
         ColumnVector expected1 = ColumnVector.fromBytes((byte)4, (byte)3, (byte)8);
         ColumnVector expected2 = ColumnVector.fromBytes((byte)100);
         ColumnVector expected3 = ColumnVector.fromBytes((byte)-23)) {
      TableTest.assertColumnsAreEqual(expected1, byteColumnVector1);
      TableTest.assertColumnsAreEqual(expected2, byteColumnVector2);
      TableTest.assertColumnsAreEqual(expected3, byteColumnVector3);
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (Builder builder = HostColumnVector.builder(DType.INT8, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append((byte) 2).appendNull().append((byte) 5, (byte) 4).build());
    }
  }

  @Test
  void testAppendVector() {
    Random random = new Random(192312989128L);
    for (int dstSize = 1; dstSize <= 100; dstSize++) {
      for (int dstPrefilledSize = 0; dstPrefilledSize < dstSize; dstPrefilledSize++) {
        final int srcSize = dstSize - dstPrefilledSize;
        for (int sizeOfDataNotToAdd = 0; sizeOfDataNotToAdd <= dstPrefilledSize; sizeOfDataNotToAdd++) {
          try (Builder dst = HostColumnVector.builder(DType.INT8, dstSize);
               HostColumnVector src = HostColumnVector.build(DType.INT8, srcSize, (b) -> {
                 for (int i = 0; i < srcSize; i++) {
                   if (random.nextBoolean()) {
                     b.appendNull();
                   } else {
                     b.append((byte) random.nextInt());
                   }
                 }
               });
               Builder gtBuilder = HostColumnVector.builder(DType.INT8, dstPrefilledSize)) {
            assertEquals(dstSize, srcSize + dstPrefilledSize);
            //add the first half of the prefilled list
            for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
              if (random.nextBoolean()) {
                dst.appendNull();
                gtBuilder.appendNull();
              } else {
                byte a = (byte) random.nextInt();
                dst.append(a);
                gtBuilder.append(a);
              }
            }
            // append the src vector
            dst.append(src);
            try (HostColumnVector dstVector = dst.build();
                 HostColumnVector gt = gtBuilder.build()) {
              for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
                assertEquals(gt.isNull(i), dstVector.isNull(i));
                if (!gt.isNull(i)) {
                  assertEquals(gt.getByte(i), dstVector.getByte(i));
                }
              }
              for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                assertEquals(src.isNull(j), dstVector.isNull(i));
                if (!src.isNull(j)) {
                  assertEquals(src.getByte(j), dstVector.getByte(i));
                }
              }
              if (dstVector.hasValidityVector()) {
                long maxIndex =
                    BitVectorHelper.getValidityAllocationSizeInBytes(dstVector.getRowCount()) * 8;
                for (long i = dstSize - sizeOfDataNotToAdd; i < maxIndex; i++) {
                  assertFalse(dstVector.isNullExtendedRange(i));
                }
              }
            }
          }
        }
      }
    }
  }
}
