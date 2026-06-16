/*
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.Builder;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class IntColumnVectorTest extends CudfTestBase {

  @Test
  public void testCreateColumnVectorBuilder() {
    try (ColumnVector intColumnVector = ColumnVector.build(DType.INT32, 3, (b) -> b.append(1))) {
      assertFalse(intColumnVector.hasNulls());
    }
    try (ColumnVector intColumnVector = ColumnBuilderHelper.buildOnDevice(
        new HostColumnVector.BasicType(true, DType.INT32), 3, (b) -> b.append(1))) {
      assertFalse(intColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertFalse(cv.hasNulls());
      assertEquals(cv.getInt(0), 2);
      assertEquals(cv.getInt(1), 3);
      assertEquals(cv.getInt(2), 5);
    };
    try (HostColumnVector cv = HostColumnVector.fromInts(2, 3, 5)) {
      verify.accept(cv);
    }
    try (HostColumnVector cv = ColumnBuilderHelper.fromInts(true, 2, 3, 5)) {
      verify.accept(cv);
    }
  }

  @Test
  public void testUnsignedArrayAllocation() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertFalse(cv.hasNulls());
      assertEquals(0xfedcba98L, Integer.toUnsignedLong(cv.getInt(0)));
      assertEquals(0x80000000L, Integer.toUnsignedLong(cv.getInt(1)));
      assertEquals(5, Integer.toUnsignedLong(cv.getInt(2)));
    };
    try (HostColumnVector cv = HostColumnVector.fromUnsignedInts(0xfedcba98, 0x80000000, 5)) {
      verify.accept(cv);
    }
    try (HostColumnVector cv = ColumnBuilderHelper.fromInts(false, 0xfedcba98, 0x80000000, 5)) {
      verify.accept(cv);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertThrows(AssertionError.class, () -> cv.getInt(3));
      assertFalse(cv.hasNulls());
    };
    try (HostColumnVector icv = HostColumnVector.fromInts(2, 3, 5)) {
      verify.accept(icv);
    }
    try (HostColumnVector icv = ColumnBuilderHelper.fromInts(true, 2, 3, 5)) {
      verify.accept(icv);
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertFalse(cv.hasNulls());
      assertThrows(AssertionError.class, () -> cv.getInt(-1));
    };
    try (HostColumnVector icv = HostColumnVector.fromInts(2, 3, 5)) {
      verify.accept(icv);
    }
    try (HostColumnVector icv = ColumnBuilderHelper.fromInts(true, 2, 3, 5)) {
      verify.accept(icv);
    }
  }

  @Test
  public void testAddingNullValues() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertTrue(cv.hasNulls());
      assertEquals(2, cv.getNullCount());
      for (int i = 0; i < 6; i++) {
        assertFalse(cv.isNull(i));
      }
      assertTrue(cv.isNull(6));
      assertTrue(cv.isNull(7));
    };
    try (HostColumnVector cv = HostColumnVector.fromBoxedInts(2, 3, 4, 5, 6, 7, null, null)) {
      verify.accept(cv);
    }
    try (HostColumnVector cv = ColumnBuilderHelper.fromBoxedInts(true, 2, 3, 4, 5, 6, 7, null, null)) {
      verify.accept(cv);
    }
  }

  @Test
  public void testAddingUnsignedNullValues() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertTrue(cv.hasNulls());
      assertEquals(2, cv.getNullCount());
      for (int i = 0; i < 6; i++) {
        assertFalse(cv.isNull(i));
      }
      assertEquals(0xfedbca98L, Integer.toUnsignedLong(cv.getInt(4)));
      assertEquals(0x80000000L, Integer.toUnsignedLong(cv.getInt(5)));
      assertTrue(cv.isNull(6));
      assertTrue(cv.isNull(7));
    };
    try (HostColumnVector cv = HostColumnVector.fromBoxedUnsignedInts(
            2, 3, 4, 5, 0xfedbca98, 0x80000000, null, null)) {
      verify.accept(cv);
    }
    try (HostColumnVector cv = ColumnBuilderHelper.fromBoxedInts(false,
        2, 3, 4, 5, 0xfedbca98, 0x80000000, null, null)) {
      verify.accept(cv);
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (Builder builder = HostColumnVector.builder(DType.INT32, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(2).appendNull().appendArray(new int[]{5, 4}).build());
    }
  }

  @Test
  public void testCastToInt() {
    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(new double[]{4.3, 3.8, 8});
         ColumnVector shortColumnVector = ColumnVector.fromShorts(new short[]{100});
         ColumnVector intColumnVector1 = doubleColumnVector.asInts();
         ColumnVector expected1 = ColumnVector.fromInts(4, 3, 8);
         ColumnVector intColumnVector2 = shortColumnVector.asInts();
         ColumnVector expected2 = ColumnVector.fromInts(100)) {
      AssertUtils.assertColumnsAreEqual(expected1, intColumnVector1);
      AssertUtils.assertColumnsAreEqual(expected2, intColumnVector2);
    }
  }

  @Test
  void testAppendVector() {
    Random random = new Random(192312989128L);
    for (int dstSize = 1; dstSize <= 100; dstSize++) {
      for (int dstPrefilledSize = 0; dstPrefilledSize < dstSize; dstPrefilledSize++) {
        final int srcSize = dstSize - dstPrefilledSize;
        for (int sizeOfDataNotToAdd = 0; sizeOfDataNotToAdd <= dstPrefilledSize; sizeOfDataNotToAdd++) {
          try (Builder dst = HostColumnVector.builder(DType.INT32, dstSize);
               HostColumnVector src = HostColumnVector.build(DType.INT32, srcSize, (b) -> {
                 for (int i = 0; i < srcSize; i++) {
                   if (random.nextBoolean()) {
                     b.appendNull();
                   } else {
                     b.append(random.nextInt());
                   }
                 }
               });
               Builder gtBuilder = HostColumnVector.builder(DType.INT32,
                   dstPrefilledSize)) {
            assertEquals(dstSize, srcSize + dstPrefilledSize);
            //add the first half of the prefilled list
            for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
              if (random.nextBoolean()) {
                dst.appendNull();
                gtBuilder.appendNull();
              } else {
                int a = random.nextInt();
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
                  assertEquals(gt.getInt(i), dstVector.getInt(i));
                }
              }
              for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                assertEquals(src.isNull(j), dstVector.isNull(i));
                if (!src.isNull(j)) {
                  assertEquals(src.getInt(j), dstVector.getInt(i));
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
