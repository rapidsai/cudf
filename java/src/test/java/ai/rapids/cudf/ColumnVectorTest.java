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

import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class ColumnVectorTest {
  @Test
  void testRefCountLeak() throws InterruptedException {
    assumeTrue(Boolean.getBoolean("ai.rapids.cudf.flaky-tests-enabled"));
    long expectedLeakCount = MemoryCleaner.leakCount.get() + 1;
    ColumnVector.fromInts(1, 2, 3);
    long maxTime = System.currentTimeMillis() + 10_000;
    long leakNow;
    do {
      System.gc();
      Thread.sleep(50);
      leakNow = MemoryCleaner.leakCount.get();
    } while (leakNow != expectedLeakCount && System.currentTimeMillis() < maxTime);
    assertEquals(expectedLeakCount, MemoryCleaner.leakCount.get());
  }

  @Test
  void testConcatTypeError() {
    try (ColumnVector v0 = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromFloats(5.0f, 6.0f)) {
      assertThrows(CudfException.class, () -> ColumnVector.concatenate(v0, v1));
    }
  }

  @Test
  void testConcatNoNulls() {
    try (ColumnVector v0 = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromInts(5, 6, 7);
         ColumnVector v2 = ColumnVector.fromInts(8, 9);
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2)) {
      v.ensureOnHost();
      assertEquals(9, v.getRowCount());
      assertFalse(v.hasNulls());
      assertFalse(v.hasValidityVector());
      for (int i = 0; i < 9; ++i) {
        assertEquals(i + 1, v.getInt(i), "at index " + i);
      }
    }
  }

  @Test
  void testConcatWithNulls() {
    try (ColumnVector v0 = ColumnVector.fromDoubles(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromDoubles(5, 6, 7);
         ColumnVector v2 = ColumnVector.fromBoxedDoubles(null, 9.0);
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2)) {
      v.ensureOnHost();
      assertEquals(9, v.getRowCount());
      assertTrue(v.hasNulls());
      assertTrue(v.hasValidityVector());
      for (int i = 0; i < 9; ++i) {
        if (i != 7) {
          assertEquals(i + 1, v.getDouble(i), "at index " + i);
        } else {
          assertTrue(v.isNull(i), "at index " + i);
        }
      }
    }
  }

  @Test
  void isNotNullTestEmptyColumn() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v = ColumnVector.fromBoxedInts();
         ColumnVector expected = ColumnVector.fromBoxedBooleans(); 
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void isNotNullTest() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v = ColumnVector.fromBoxedInts(1, 2, null, 4, null, 6);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, false, true, false, true);
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void isNotNullTestAllNulls() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v = ColumnVector.fromBoxedInts(null, null, null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false, false, false);
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void isNotNullTestAllNotNulls() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v = ColumnVector.fromBoxedInts(1,2,3,4,5,6);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true, true, true);
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void isNullTest() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v = ColumnVector.fromBoxedInts(1, 2, null, 4, null, 6);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, false, true, false);
         ColumnVector result = v.isNull()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testFromScalarProducesEmptyColumn() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromScalar(Scalar.fromInt(1), 0);
         ColumnVector expected = ColumnVector.fromBoxedInts()) {
      assertColumnsAreEqual(input, expected);
    }
  }

  @Test
  void testFromScalarFloat() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromScalar(Scalar.fromFloat(1.123f), 4);
         ColumnVector expected = ColumnVector.fromBoxedFloats(1.123f, 1.123f, 1.123f, 1.123f)) {
      assertColumnsAreEqual(input, expected);
    }
  }

  @Test
  void testFromScalarInteger() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromScalar(Scalar.fromNull(DType.INT32), 6);
         ColumnVector expected = ColumnVector.fromBoxedInts(null, null, null, null, null, null)) {
      assertEquals(input.getNullCount(), expected.getNullCount());
      assertColumnsAreEqual(input, expected);
    }
  }

  @Test
  void testFromScalarStringThrows() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    assertThrows(IllegalArgumentException.class, () ->
      ColumnVector.fromScalar(Scalar.fromString("test"), 1));
  }

  @Test
  void testReplaceEmptyColumn() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedBooleans();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         ColumnVector result = input.replaceNulls(Scalar.fromBool(false))) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullBoolsWithAllNulls() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedBooleans(null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false);
         ColumnVector result = input.replaceNulls(Scalar.fromBool(false))) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceSomeNullBools() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedBooleans(false, null, null, false);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false);
         ColumnVector result = input.replaceNulls(Scalar.fromBool(true))) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullIntegersWithAllNulls() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedInts(null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedInts(0, 0, 0, 0);
         ColumnVector result = input.replaceNulls(Scalar.fromInt(0))) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceSomeNullIntegers() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null);
         ColumnVector expected = ColumnVector.fromBoxedInts(1, 2, 999, 4, 999);
         ColumnVector result = input.replaceNulls(Scalar.fromInt(999))) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullsFailsOnTypeMismatch() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null)) {
      assertThrows(CudfException.class, () -> {
        long nativePtr = Cudf.replaceNulls(input, Scalar.fromBool(true));
        if (nativePtr != 0) {
          new ColumnVector(nativePtr).close();
        }
      });
    }
  }

  @Test
  void testReplaceNullsFailsOnNullScalar() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null)) {
      assertThrows(CudfException.class, () -> {
        long nativePtr = Cudf.replaceNulls(input, Scalar.fromNull(input.getType()));
        if (nativePtr != 0) {
          new ColumnVector(nativePtr).close();
        }
      });
    }
  }
}
