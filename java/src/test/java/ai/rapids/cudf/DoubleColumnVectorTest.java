/*
 *
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

public class DoubleColumnVectorTest extends CudfTestBase {

  @Test
  public void testCreateColumnVectorBuilder() {
    try (ColumnVector doubleColumnVector = ColumnVector.build(DType.FLOAT64, 3,
        (b) -> b.append(1.0))) {
      assertFalse(doubleColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertFalse(cv.hasNulls());
      assertEqualsWithinPercentage(cv.getDouble(0), 2.1, 0.01);
      assertEqualsWithinPercentage(cv.getDouble(1), 3.02, 0.01);
      assertEqualsWithinPercentage(cv.getDouble(2), 5.003, 0.001);
    };
    try (HostColumnVector dcv = HostColumnVector.fromDoubles(2.1, 3.02, 5.003)) {
      verify.accept(dcv);
    }
    try (HostColumnVector dcv = ColumnBuilderHelper.fromDoubles(2.1, 3.02, 5.003)) {
      verify.accept(dcv);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertThrows(AssertionError.class, () -> cv.getDouble(3));
      assertFalse(cv.hasNulls());
    };
    try (HostColumnVector dcv = HostColumnVector.fromDoubles(2.1, 3.02, 5.003)) {
      verify.accept(dcv);
    }
    try (HostColumnVector dcv = ColumnBuilderHelper.fromDoubles(2.1, 3.02, 5.003)) {
      verify.accept(dcv);
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    Consumer<HostColumnVector> verify = (cv) -> {
      assertFalse(cv.hasNulls());
      assertThrows(AssertionError.class, () -> cv.getDouble(-1));
    };
    try (HostColumnVector dcv = HostColumnVector.fromDoubles(2.1, 3.02, 5.003)) {
      verify.accept(dcv);
    }
    try (HostColumnVector dcv = ColumnBuilderHelper.fromDoubles(2.1, 3.02, 5.003)) {
      verify.accept(dcv);
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
    try (HostColumnVector dcv =
             HostColumnVector.fromBoxedDoubles(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, null, null)) {
      verify.accept(dcv);
    }
    try (HostColumnVector dcv = ColumnBuilderHelper.fromBoxedDoubles(
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, null, null)) {
      verify.accept(dcv);
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (Builder builder = HostColumnVector.builder(DType.FLOAT64, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(2.1).appendNull().appendArray(new double[]{5.003, 4.0}).build());
    }
  }

  @Test
  void testAppendVector() {
    Random random = new Random(192312989128L);
    for (int dstSize = 1; dstSize <= 100; dstSize++) {
      for (int dstPrefilledSize = 0; dstPrefilledSize < dstSize; dstPrefilledSize++) {
        final int srcSize = dstSize - dstPrefilledSize;
        for (int sizeOfDataNotToAdd = 0; sizeOfDataNotToAdd <= dstPrefilledSize; sizeOfDataNotToAdd++) {
          try (Builder dst = HostColumnVector.builder(DType.FLOAT64, dstSize);
               HostColumnVector src = HostColumnVector.build(DType.FLOAT64, srcSize, (b) -> {
                 for (int i = 0; i < srcSize; i++) {
                   if (random.nextBoolean()) {
                     b.appendNull();
                   } else {
                     b.append(random.nextDouble());
                   }
                 }
               });
               Builder gtBuilder = HostColumnVector.builder(DType.FLOAT64, dstPrefilledSize)) {
            assertEquals(dstSize, srcSize + dstPrefilledSize);
            //add the first half of the prefilled list
            for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
              if (random.nextBoolean()) {
                dst.appendNull();
                gtBuilder.appendNull();
              } else {
                double a = random.nextDouble();
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
                  assertEquals(gt.getDouble(i), dstVector.getDouble(i));
                }
              }
              for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                assertEquals(src.isNull(j), dstVector.isNull(i));
                if (!src.isNull(j)) {
                  assertEquals(src.getDouble(j), dstVector.getDouble(i));
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
