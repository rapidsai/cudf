
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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class UnaryOpTest {
  private static final Double[] DOUBLES_1 = new Double[]{1.0, 10.0, -100.1, 5.3, 50.0, 100.0, null};
  private static final Integer[] INTS_1 = new Integer[]{1, 10, -100, 5, 50, 100, null};
  private static final String[] STRINGS_1 = new String[]{"1", "10", "-100", "5", "50", "100", null};

  // These tests are not for the correctness of the underlying implementation, but really just
  // plumbing

  @Test
  public void testSin() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.sin();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.sin(1.0), Math.sin(10.0),
             Math.sin(-100.1), Math.sin(5.3), Math.sin(50.0), Math.sin(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testCos() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.cos();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.cos(1.0), Math.cos(10.0),
             Math.cos(-100.1), Math.cos(5.3), Math.cos(50.0), Math.cos(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testTan() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.tan();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.tan(1.0), Math.tan(10.0),
             Math.tan(-100.1), Math.tan(5.3), Math.tan(50.0), Math.tan(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testArcsin() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.arcsin();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.asin(1.0), Math.asin(10.0),
             Math.asin(-100.1), Math.asin(5.3), Math.asin(50.0), Math.asin(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testArccos() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.arccos();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.acos(1.0), Math.acos(10.0),
             Math.acos(-100.1), Math.acos(5.3), Math.acos(50.0), Math.acos(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testArctan() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.arctan();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.atan(1.0), Math.atan(10.0),
             Math.atan(-100.1), Math.atan(5.3), Math.atan(50.0), Math.atan(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testExp() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.exp();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.exp(1.0), Math.exp(10.0),
             Math.exp(-100.1), Math.exp(5.3), Math.exp(50.0), Math.exp(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testLog() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.log();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.log(1.0), Math.log(10.0),
             Math.log(-100.1), Math.log(5.3), Math.log(50.0), Math.log(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testSqrt() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.sqrt();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.sqrt(1.0), Math.sqrt(10.0),
             Math.sqrt(-100.1), Math.sqrt(5.3), Math.sqrt(50.0), Math.sqrt(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testCeil() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.ceil();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.ceil(1.0), Math.ceil(10.0),
             Math.ceil(-100.1), Math.ceil(5.3), Math.ceil(50.0), Math.ceil(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testFloor() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.floor();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.floor(1.0), Math.floor(10.0),
             Math.floor(-100.1), Math.floor(5.3), Math.floor(50.0), Math.floor(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testAbs() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.abs();
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.abs(1.0), Math.abs(10.0),
             Math.abs(-100.1), Math.abs(5.3), Math.abs(50.0), Math.abs(100.0), null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testBitInvert() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector answer = icv.bitInvert();
         ColumnVector expected = ColumnVector.fromBoxedInts(~1, ~10, ~-100, ~5, ~50, ~100, null)) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  // String to string cat conversion has more to do with correctness as we wrote that all ourselves
  @Test
  public void testStringCastFullCircle() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector origStr = ColumnVector.fromStrings(STRINGS_1);
         ColumnVector origCat = ColumnVector.categoryFromStrings(STRINGS_1);
         ColumnVector cat = origStr.asStringCategories();
         ColumnVector str = origCat.asStrings();
         ColumnVector catAgain = str.asStringCategories();
         ColumnVector strAgain = cat.asStrings()) {
      assertColumnsAreEqual(origCat, cat);
      assertColumnsAreEqual(origStr, str);
      assertColumnsAreEqual(origCat, catAgain);
      assertColumnsAreEqual(origStr, strAgain);
    }
  }
}
