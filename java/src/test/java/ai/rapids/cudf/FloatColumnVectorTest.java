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

public class FloatColumnVectorTest {

  @Test
  public void testCreateColumnVectorBuilder() {
    try (ColumnVector floatColumnVector = ColumnVector.build(DType.FLOAT32, 3,
        (b) -> b.append(1.0f))) {
      assertFalse(floatColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    try (ColumnVector floatColumnVector = ColumnVector.fromFloats(2.1f, 3.02f, 5.003f)) {
      assertFalse(floatColumnVector.hasNulls());
      assertEquals(floatColumnVector.getFloat(0), 2.1, 0.01);
      assertEquals(floatColumnVector.getFloat(1), 3.02, 0.01);
      assertEquals(floatColumnVector.getFloat(2), 5.003, 0.001);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    try (ColumnVector floatColumnVector = ColumnVector.fromFloats(2.1f, 3.02f, 5.003f)) {
      assertThrows(AssertionError.class, () -> floatColumnVector.getFloat(3));
      assertFalse(floatColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    try (ColumnVector floatColumnVector = ColumnVector.fromFloats(2.1f, 3.02f, 5.003f)) {
      assertFalse(floatColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> floatColumnVector.getFloat(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    try (ColumnVector cv = ColumnVector.fromBoxedFloats(
        new Float[]{2f, 3f, 4f, 5f, 6f, 7f, null, null})) {
      assertTrue(cv.hasNulls());
      assertEquals(2, cv.getNullCount());
      for (int i = 0; i < 6; i++) {
        assertFalse(cv.isNull(i));
      }
      assertTrue(cv.isNull(6));
      assertTrue(cv.isNull(7));
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.FLOAT32, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(2.1f).appendNull().appendArray(5.003f, 4.0f).build());
    }
  }

  @Test
  public void testCastToFloat() {
    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(new double[]{4.3, 3.8, 8});
         ColumnVector shortColumnVector = ColumnVector.fromShorts(new short[]{100});
         ColumnVector floatColumnVector1 = doubleColumnVector.asFloats();
         ColumnVector floatColumnVector2 = shortColumnVector.asFloats()) {
      floatColumnVector1.ensureOnHost();
      floatColumnVector2.ensureOnHost();
      assertEquals(4.3, floatColumnVector1.getFloat(0), 0.001);
      assertEquals(3.8, floatColumnVector1.getFloat(1), 0.001);
      assertEquals(8, floatColumnVector1.getFloat(2));
      assertEquals(100, floatColumnVector2.getFloat(0));
    }
  }

  @Test
  void testAppendVector() {
    Random random = new Random(192312989128L);
    for (int dstSize = 1; dstSize <= 100; dstSize++) {
      for (int dstPrefilledSize = 0; dstPrefilledSize < dstSize; dstPrefilledSize++) {
        final int srcSize = dstSize - dstPrefilledSize;
        for (int sizeOfDataNotToAdd = 0; sizeOfDataNotToAdd <= dstPrefilledSize; sizeOfDataNotToAdd++) {
          try (ColumnVector.Builder dst = ColumnVector.builder(DType.FLOAT32, dstSize);
               ColumnVector src = ColumnVector.buildOnHost(DType.FLOAT32, srcSize, (b) -> {
                 for (int i = 0; i < srcSize; i++) {
                   if (random.nextBoolean()) {
                     b.appendNull();
                   } else {
                     b.append(random.nextFloat());
                   }
                 }
               });
               ColumnVector.Builder gtBuilder = ColumnVector.builder(DType.FLOAT32,
                   dstPrefilledSize)) {
            assertEquals(dstSize, srcSize + dstPrefilledSize);
            //add the first half of the prefilled list
            for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
              if (random.nextBoolean()) {
                dst.appendNull();
                gtBuilder.appendNull();
              } else {
                float a = random.nextFloat();
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
                  assertEquals(gt.getFloat(i), dstVector.getFloat(i));
                }
              }
              for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                assertEquals(src.isNull(j), dstVector.isNull(i));
                if (!src.isNull(j)) {
                  assertEquals(src.getFloat(j), dstVector.getFloat(i));
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
