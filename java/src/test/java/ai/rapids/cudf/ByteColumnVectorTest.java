
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

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ByteColumnVectorTest extends CudfTestBase {

  @Test
  public void testCreateColumnVectorBuilder() {
    try (ColumnVector shortColumnVector = ColumnVector.build(DType.INT8, 3,
        (b) -> b.append((byte) 1))) {
      assertFalse(shortColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    try (ColumnVector byteColumnVector = ColumnVector.fromBytes(new byte[]{2, 3, 5})) {
      assertFalse(byteColumnVector.hasNulls());
      assertEquals(byteColumnVector.getByte(0), 2);
      assertEquals(byteColumnVector.getByte(1), 3);
      assertEquals(byteColumnVector.getByte(2), 5);
    }
  }

  @Test
  public void testAppendRepeatingValues() {
    try (ColumnVector byteColumnVector = ColumnVector.build(DType.INT8, 3,
        (b) -> b.append((byte) 2, (long) 3))) {
      assertFalse(byteColumnVector.hasNulls());
      assertEquals(byteColumnVector.getByte(0), 2);
      assertEquals(byteColumnVector.getByte(1), 2);
      assertEquals(byteColumnVector.getByte(2), 2);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    try (ColumnVector byteColumnVector = ColumnVector.fromBytes(new byte[]{2, 3, 5})) {
      assertThrows(AssertionError.class, () -> byteColumnVector.getByte(3));
      assertFalse(byteColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    try (ColumnVector byteColumnVector = ColumnVector.fromBytes(new byte[]{2, 3, 5})) {
      assertFalse(byteColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> byteColumnVector.getByte(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    try (ColumnVector byteColumnVector = ColumnVector.fromBoxedBytes(
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
  public void testCastToByte() {
    final int[] DATES = {17897}; //Jan 01, 2019

    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(new double[]{4.3, 3.8, 8});
         ColumnVector shortColumnVector = ColumnVector.fromShorts(new short[]{100});
         ColumnVector dateColumnVector = ColumnVector.datesFromInts(DATES);
         ColumnVector byteColumnVector1 = doubleColumnVector.asBytes();
         ColumnVector byteColumnVector2 = shortColumnVector.asBytes();
         ColumnVector byteColumnVector3 = dateColumnVector.asBytes()) {
      byteColumnVector1.ensureOnHost();
      byteColumnVector2.ensureOnHost();
      byteColumnVector3.ensureOnHost();
      assertEquals(byteColumnVector1.getByte(0), 4);
      assertEquals(byteColumnVector1.getByte(1), 3);
      assertEquals(byteColumnVector1.getByte(2), 8);
      assertEquals(byteColumnVector2.getByte(0), 100);
      assertEquals(byteColumnVector3.getByte(0), -23);
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.INT8, 3)) {
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
          try (ColumnVector.Builder dst = ColumnVector.builder(DType.INT8, dstSize);
               ColumnVector src = ColumnVector.buildOnHost(DType.INT8, srcSize, (b) -> {
                 for (int i = 0; i < srcSize; i++) {
                   if (random.nextBoolean()) {
                     b.appendNull();
                   } else {
                     b.append((byte) random.nextInt());
                   }
                 }
               });
               ColumnVector.Builder gtBuilder = ColumnVector.builder(DType.INT8,
                   dstPrefilledSize)) {
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
            try (ColumnVector dstVector = dst.buildOnHost();
                 ColumnVector gt = gtBuilder.buildOnHost()) {
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
