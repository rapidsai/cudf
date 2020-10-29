/*
 *  Copyright (c) 2020, NVIDIA CORPORATION.
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

import java.math.BigDecimal;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class DecimalColumnVectorTest extends CudfTestBase {
  private final Random rdSeed = new Random();

  private final BigDecimal[] decimal32Zoo = new BigDecimal[]{
    BigDecimal.valueOf(rdSeed.nextInt(1000000000), 4),
    BigDecimal.valueOf(rdSeed.nextInt(1000000000), 4),
    BigDecimal.valueOf(rdSeed.nextInt(1000000000), 4),
  };

  private final BigDecimal[] decimal64Zoo = new BigDecimal[]{
    BigDecimal.valueOf(rdSeed.nextLong(), 10),
    BigDecimal.valueOf(rdSeed.nextLong(), 10),
    null,
    BigDecimal.valueOf(rdSeed.nextLong(), 10),
  };

  @Test
  public void testCreateColumnVectorBuilder() {
    try (ColumnVector decColumnVector = ColumnVector.build(DType.create(DType.DTypeEnum.DECIMAL32, -5), 3,
        (b) -> b.append(BigDecimal.valueOf(123456789, 5)))) {
      assertFalse(decColumnVector.hasNulls());
    }
    try (ColumnVector decColumnVector = ColumnVector.build(DType.create(DType.DTypeEnum.DECIMAL64, -10), 3,
        (b) -> b.append(BigDecimal.valueOf(1023040506070809L, 10)))) {
      assertFalse(decColumnVector.hasNulls());
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    try (HostColumnVector decColumnVector = HostColumnVector.fromBigDecimals(decimal32Zoo)) {
      assertThrows(AssertionError.class, () -> decColumnVector.getBigDecimal(3));
      assertFalse(decColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    try (HostColumnVector doubleColumnVector = HostColumnVector.fromBigDecimals(decimal32Zoo)) {
      assertFalse(doubleColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> doubleColumnVector.getBigDecimal(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    try (HostColumnVector cv = HostColumnVector.fromBigDecimals(decimal64Zoo)) {
      assertTrue(cv.hasNulls());
      assertEquals(1, cv.getNullCount());
      assertFalse(cv.isNull(0));
      assertFalse(cv.isNull(1));
      assertTrue(cv.isNull(2));
      assertFalse(cv.isNull(3));
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (Builder builder = HostColumnVector.builder(DType.create(DType.DTypeEnum.DECIMAL32, 4), 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(decimal32Zoo[0]).appendNull().appendBoxed(decimal32Zoo[1], decimal32Zoo[2]).build());
    }
  }

  @Test
  public void testDecimalValidation() {
    assertThrows(AssertionError.class,
        () -> HostColumnVector.fromBigDecimals(BigDecimal.valueOf(123, 1), BigDecimal.valueOf(123, 2)));
    assertThrows(AssertionError.class,
        () -> HostColumnVector.fromBigDecimals(new BigDecimal("12345678901234567890")));
  }

  @Test
  public void testDecimalSpecifics() {
    try (HostColumnVector cv = HostColumnVector.fromBigDecimals(decimal32Zoo)) {
      assertEquals(DType.DTypeEnum.DECIMAL32, cv.getType().typeId);
      assertEquals(-4, cv.getType().getScale());
      assertFalse(cv.hasNulls());
      assertEquals(decimal32Zoo[0], cv.getBigDecimal(0));
      assertEquals(decimal32Zoo[1], cv.getBigDecimal(1));
      assertEquals(decimal32Zoo[2], cv.getBigDecimal(2));
    }
    try (HostColumnVector cv = HostColumnVector.fromBigDecimals(decimal64Zoo)) {
      assertEquals(DType.DTypeEnum.DECIMAL64, cv.getType().typeId);
      assertEquals(-10, cv.getType().getScale());
      assertTrue(cv.hasNulls());
      assertEquals(decimal64Zoo[0], cv.getBigDecimal(0));
      assertEquals(decimal64Zoo[1], cv.getBigDecimal(1));
      assertThrows(AssertionError.class, () -> cv.getBigDecimal(2));
      assertEquals(decimal64Zoo[3], cv.getBigDecimal(3));
    }
  }

  @Test
  public void testAppendVector() {
    Random random = new Random(192312989128L);
    for (DType decType : new DType[]{
        DType.create(DType.DTypeEnum.DECIMAL32, -6),
        DType.create(DType.DTypeEnum.DECIMAL64, -10)}) {
      for (int dstSize = 1; dstSize <= 100; dstSize++) {
        for (int dstPrefilledSize = 0; dstPrefilledSize < dstSize; dstPrefilledSize++) {
          final int srcSize = dstSize - dstPrefilledSize;
          for (int sizeOfDataNotToAdd = 0; sizeOfDataNotToAdd <= dstPrefilledSize; sizeOfDataNotToAdd++) {
            try (Builder dst = HostColumnVector.builder(decType, dstSize);
                 HostColumnVector src = HostColumnVector.build(decType, srcSize, (b) -> {
                   for (int i = 0; i < srcSize; i++) {
                     if (random.nextBoolean()) {
                       b.appendNull();
                     } else {
                       b.append(BigDecimal.valueOf(random.nextInt(), -decType.getScale()));
                     }
                   }
                 });
                 Builder gtBuilder = HostColumnVector.builder(decType, dstPrefilledSize)) {
              assertEquals(dstSize, srcSize + dstPrefilledSize);
              //add the first half of the prefilled list
              for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
                if (random.nextBoolean()) {
                  dst.appendNull();
                  gtBuilder.appendNull();
                } else {
                  BigDecimal a = BigDecimal.valueOf(random.nextInt(), -decType.getScale());
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
                    assertEquals(gt.getBigDecimal(i), dstVector.getBigDecimal(i));
                  }
                }
                for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                  assertEquals(src.isNull(j), dstVector.isNull(i));
                  if (!src.isNull(j)) {
                    assertEquals(src.getBigDecimal(j), dstVector.getBigDecimal(i));
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
}
