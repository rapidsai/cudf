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

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;

class ReplaceNullsTest {

  @Test
  void testReplaceEmptyColumn() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedBooleans();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         ColumnVector result = new ColumnVector(Cudf.replaceNulls(input, Scalar.fromBool(false)))) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullBoolsWithAllNulls() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedBooleans(null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false);
         ColumnVector result = new ColumnVector(Cudf.replaceNulls(input, Scalar.fromBool(false)))) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceSomeNullBools() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedBooleans(false, null, null, false);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false);
         ColumnVector result = new ColumnVector(Cudf.replaceNulls(input, Scalar.fromBool(true)))) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullIntegersWithAllNulls() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedInts(null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedInts(0, 0, 0, 0);
         ColumnVector result = new ColumnVector(Cudf.replaceNulls(input, Scalar.fromInt(0)))) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceSomeNullIntegers() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null);
         ColumnVector expected = ColumnVector.fromBoxedInts(1, 2, 999, 4, 999);
         ColumnVector result = new ColumnVector(Cudf.replaceNulls(input, Scalar.fromInt(999)))) {
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
