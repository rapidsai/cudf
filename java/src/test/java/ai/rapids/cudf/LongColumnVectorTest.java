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

public class LongColumnVectorTest extends CudfTestBase {

  @Test
  public void testCreateColumnVectorBuilder() {
    try (ColumnVector longColumnVector = ColumnVector.build(DType.INT64, 3, (b) -> b.append(1L))) {
      assertFalse(longColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertFalse(cv.hasNulls());
      assertEquals(cv.getLong(0), 2);
      assertEquals(cv.getLong(1), 3);
      assertEquals(cv.getLong(2), 5);
    };
    try (HostColumnVector lcv = HostColumnVector.fromLongs(2L, 3L, 5L)) {
      verify.accept(lcv);
    }
    try (HostColumnVector lcv = ColumnBuilderHelper.fromLongs(true,2L, 3L, 5L)) {
      verify.accept(lcv);
    }
  }

  @Test
  public void testUnsignedArrayAllocation() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertFalse(cv.hasNulls());
      assertEquals(Long.toUnsignedString(0xfedcba9876543210L),
          Long.toUnsignedString(cv.getLong(0)));
      assertEquals(Long.toUnsignedString(0x8000000000000000L),
          Long.toUnsignedString(cv.getLong(1)));
      assertEquals(5L, cv.getLong(2));
    };
    try (HostColumnVector lcv = HostColumnVector.fromUnsignedLongs(
        0xfedcba9876543210L, 0x8000000000000000L, 5L)) {
      verify.accept(lcv);
    }
    try (HostColumnVector lcv = ColumnBuilderHelper.fromLongs(false,
        0xfedcba9876543210L, 0x8000000000000000L, 5L)) {
      verify.accept(lcv);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertThrows(AssertionError.class, () -> cv.getLong(3));
      assertFalse(cv.hasNulls());
    };
    try (HostColumnVector lcv = HostColumnVector.fromLongs(2L, 3L, 5L)) {
      verify.accept(lcv);
    }
    try (HostColumnVector lcv = ColumnBuilderHelper.fromLongs(true, 2L, 3L, 5L)) {
      verify.accept(lcv);
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertFalse(cv.hasNulls());
      assertThrows(AssertionError.class, () -> cv.getLong(-1));
    };
    try (HostColumnVector lcv = HostColumnVector.fromLongs(2L, 3L, 5L)) {
      verify.accept(lcv);
    }
    try (HostColumnVector lcv = ColumnBuilderHelper.fromLongs(true, 2L, 3L, 5L)) {
      verify.accept(lcv);
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
    try (HostColumnVector lcv = HostColumnVector.fromBoxedLongs(2L, 3L, 4L, 5L, 6L, 7L, null, null)) {
      verify.accept(lcv);
    }
    try (HostColumnVector lcv = ColumnBuilderHelper.fromBoxedLongs(true,
        2L, 3L, 4L, 5L, 6L, 7L, null, null)) {
      verify.accept(lcv);
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
      assertEquals(Long.toUnsignedString(0xfedcba9876543210L),
          Long.toUnsignedString(cv.getLong(4)));
      assertEquals(Long.toUnsignedString(0x8000000000000000L),
          Long.toUnsignedString(cv.getLong(5)));
      assertTrue(cv.isNull(6));
      assertTrue(cv.isNull(7));
    };
    try (HostColumnVector lcv = HostColumnVector.fromBoxedUnsignedLongs(
        2L, 3L, 4L, 5L, 0xfedcba9876543210L, 0x8000000000000000L, null, null)) {
      verify.accept(lcv);
    }
    try (HostColumnVector lcv = ColumnBuilderHelper.fromBoxedLongs(false,
        2L, 3L, 4L, 5L, 0xfedcba9876543210L, 0x8000000000000000L, null, null)) {
      verify.accept(lcv);
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (Builder builder = HostColumnVector.builder(DType.INT64, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(2L).appendNull().append(5L).append(4L).build());
    }
  }

  @Test
  void testAppendVector() {
    Random random = new Random(192312989128L);
    for (int dstSize = 1; dstSize <= 100; dstSize++) {
      for (int dstPrefilledSize = 0; dstPrefilledSize < dstSize; dstPrefilledSize++) {
        final int srcSize = dstSize - dstPrefilledSize;
        for (int sizeOfDataNotToAdd = 0; sizeOfDataNotToAdd <= dstPrefilledSize; sizeOfDataNotToAdd++) {
          try (Builder dst = HostColumnVector.builder(DType.INT64, dstSize);
               HostColumnVector src = HostColumnVector.build(DType.INT64, srcSize, (b) -> {
                 for (int i = 0; i < srcSize; i++) {
                   if (random.nextBoolean()) {
                     b.appendNull();
                   } else {
                     b.append(random.nextLong());
                   }
                 }
               });
               Builder gtBuilder = HostColumnVector.builder(DType.INT64,
                   dstPrefilledSize)) {
            assertEquals(dstSize, srcSize + dstPrefilledSize);
            //add the first half of the prefilled list
            for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
              if (random.nextBoolean()) {
                dst.appendNull();
                gtBuilder.appendNull();
              } else {
                long a = random.nextLong();
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
                  assertEquals(gt.getLong(i), dstVector.getLong(i));
                }
              }
              for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                assertEquals(src.isNull(j), dstVector.isNull(i));
                if (!src.isNull(j)) {
                  assertEquals(src.getLong(j), dstVector.getLong(i));
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
