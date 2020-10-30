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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class DecimalColumnVectorTest extends CudfTestBase {
  private static final Random rdSeed = new Random(1234);
  private static final int dec32Scale = 4;
  private static final int dec64Scale = 10;

  private static final BigDecimal[] decimal32Zoo = new BigDecimal[20];
  private static final BigDecimal[] decimal64Zoo = new BigDecimal[20];
  private static final int[] unscaledDec32Zoo = new int[decimal32Zoo.length];
  private static final long[] unscaledDec64Zoo = new long[decimal64Zoo.length];

  private final BigDecimal[] boundaryDecimal32 = new BigDecimal[]{
      new BigDecimal("999999999"), new BigDecimal("-999999999")};

  private final BigDecimal[] boundaryDecimal64 = new BigDecimal[]{
      new BigDecimal("999999999999999999"), new BigDecimal("-999999999999999999")};

  private final BigDecimal[] overflowDecimal32 = new BigDecimal[]{
      BigDecimal.valueOf(Integer.MAX_VALUE), BigDecimal.valueOf(Integer.MIN_VALUE)};

  private final BigDecimal[] overflowDecimal64 = new BigDecimal[]{
      BigDecimal.valueOf(Long.MAX_VALUE), BigDecimal.valueOf(Long.MIN_VALUE)};

  @BeforeAll
  public static void setup() {
    for (int i = 0; i < decimal32Zoo.length; i++) {
      unscaledDec32Zoo[i] = rdSeed.nextInt() / 10;
      unscaledDec64Zoo[i] = rdSeed.nextLong() / 10;
      if (rdSeed.nextBoolean()) {
        decimal32Zoo[i] = BigDecimal.valueOf(rdSeed.nextInt() / 10, dec32Scale);
      } else {
        decimal32Zoo[i] = null;
      }
      if (rdSeed.nextBoolean()) {
        decimal64Zoo[i] = BigDecimal.valueOf(rdSeed.nextLong() / 10, dec64Scale);
      } else {
        decimal64Zoo[i] = null;
      }
    }
  }

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
    try (HostColumnVector decColumnVector = HostColumnVector.fromDecimals(decimal32Zoo)) {
      assertThrows(AssertionError.class, () -> decColumnVector.getBigDecimal(decimal32Zoo.length));
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    try (HostColumnVector doubleColumnVector = HostColumnVector.fromDecimals(decimal32Zoo)) {
      assertThrows(AssertionError.class, () -> doubleColumnVector.getBigDecimal(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    try (HostColumnVector cv = HostColumnVector.fromDecimals(decimal64Zoo)) {
      for (int i = 0; i < decimal64Zoo.length; ++i) {
        assertEquals(decimal64Zoo[i] == null, cv.isNull(i));
      }
      assertEquals(Arrays.stream(decimal64Zoo).filter(Objects::isNull).count(), cv.getNullCount());
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (Builder builder = HostColumnVector.builder(DType.create(DType.DTypeEnum.DECIMAL32, dec32Scale), 3)) {
      assertThrows(AssertionError.class, () -> builder.appendBoxed(decimal32Zoo).build());
    }
    try (Builder builder = HostColumnVector.builder(DType.create(DType.DTypeEnum.DECIMAL64, dec64Scale), 3)) {
      assertThrows(AssertionError.class, () -> builder.appendUnscaledDecimalArray(unscaledDec64Zoo).build());
    }
  }

  @Test
  public void testDecimalValidation() {
    // inconsistent scales
    assertThrows(AssertionError.class,
        () -> HostColumnVector.fromDecimals(BigDecimal.valueOf(12.3), BigDecimal.valueOf(1.23)));
    // precision overflow
    assertThrows(IllegalArgumentException.class, () -> HostColumnVector.fromDecimals(overflowDecimal64));
    assertThrows(IllegalArgumentException.class, () -> {
      ColumnVector.decimalFromInts(-(DType.DECIMAL32_MAX_PRECISION + 1), unscaledDec32Zoo);
    });
    assertThrows(IllegalArgumentException.class, () -> {
      ColumnVector.decimalFromLongs(-(DType.DECIMAL64_MAX_PRECISION + 1), unscaledDec64Zoo);
    });
  }

  @Test
  public void testDecimalSpecifics() {
    DecimalColumnVectorTest.testDecimalImpl(false, dec32Scale, decimal32Zoo);
    DecimalColumnVectorTest.testDecimalImpl(true, dec64Scale, decimal64Zoo);
    DecimalColumnVectorTest.testDecimalImpl(false, 0, boundaryDecimal32);
    DecimalColumnVectorTest.testDecimalImpl(true, 0, boundaryDecimal64);
    DecimalColumnVectorTest.testUnscaledDec32Impl(dec32Scale, unscaledDec32Zoo);
    DecimalColumnVectorTest.testUnscaledDec64Impl(dec64Scale, unscaledDec64Zoo);
    // Safe max precision of Decimal32 is 9, so integers have 10 digits will be backed by DECIMAL64.
    try (ColumnVector cv = ColumnVector.fromDecimals(overflowDecimal32)) {
      assertEquals(DType.create(DType.DTypeEnum.DECIMAL64, 0), cv.getDataType());
    }
    // Create DECIMAL64 vector with small values
    try (ColumnVector cv =  ColumnVector.decimalFromLongs(0, 0L)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        assertTrue(hcv.getType().isBackedByLong());
        assertEquals(0L, hcv.getBigDecimal(0).longValue());
      }
    }
  }

  private static void testDecimalImpl(boolean isInt64, int scale, BigDecimal[] decimalZoo) {
    try (ColumnVector cv = ColumnVector.fromDecimals(decimalZoo)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        assertEquals(-scale, hcv.getType().getScale());
        assertEquals(isInt64, hcv.getType().typeId == DType.DTypeEnum.DECIMAL64);
        assertEquals(decimalZoo.length, hcv.rows);
        for (int i = 0; i < decimalZoo.length; i++) {
          assertEquals(decimalZoo[i] == null, hcv.isNull(i));
          if (decimalZoo[i] != null) {
            assertEquals(decimalZoo[i], hcv.getBigDecimal(i));
            long backValue = isInt64 ? hcv.getLong(i) : hcv.getInt(i);
            assertEquals(decimalZoo[i], BigDecimal.valueOf(backValue, scale));
          }
        }
      }
    }
  }

  private static void testUnscaledDec32Impl(int scale, int[] unscaledZoo) {
    try (ColumnVector cv = ColumnVector.decimalFromInts(-scale, unscaledZoo)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        for (int i = 0; i < unscaledZoo.length; i++) {
          assertEquals(unscaledZoo[i], hcv.getInt(i));
          assertEquals(BigDecimal.valueOf(unscaledZoo[i], scale), hcv.getBigDecimal(i));
        }
      }
    }
  }

  private static void testUnscaledDec64Impl(int scale, long[] unscaledZoo) {
    try (ColumnVector cv = ColumnVector.decimalFromLongs(-scale, unscaledZoo)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        for (int i = 0; i < unscaledZoo.length; i++) {
          assertEquals(unscaledZoo[i], hcv.getLong(i));
          assertEquals(BigDecimal.valueOf(unscaledZoo[i], scale), hcv.getBigDecimal(i));
        }
      }
    }
  }

  @Test
  public void testAppendVector() {
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
                     if (rdSeed.nextBoolean()) {
                       b.appendNull();
                     } else {
                       b.append(BigDecimal.valueOf(rdSeed.nextInt(), -decType.getScale()));
                     }
                   }
                 });
                 Builder gtBuilder = HostColumnVector.builder(decType, dstPrefilledSize)) {
              assertEquals(dstSize, srcSize + dstPrefilledSize);
              //add the first half of the prefilled list
              for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
                if (rdSeed.nextBoolean()) {
                  dst.appendNull();
                  gtBuilder.appendNull();
                } else {
                  BigDecimal a = BigDecimal.valueOf(rdSeed.nextInt(), -decType.getScale());
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
