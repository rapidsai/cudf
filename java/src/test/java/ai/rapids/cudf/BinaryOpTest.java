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

public class BinaryOpTest {
  private static final Integer[] INTS_1 = new Integer[]{1, 2, 3, 4, 5, null, 100};
  private static final Integer[] INTS_2 = new Integer[]{10, 20, 30, 40, 50, 60, 100};
  private static final Byte[] BYTES_1 = new Byte[]{-1, 7, 123, null, 50, 60, 100};
  private static final Float[] FLOATS_1 = new Float[]{1f, 10f, 100f, 5.3f, 50f, 100f, null};
  private static final Float[] FLOATS_2 = new Float[]{10f, 20f, 30f, 40f, 50f, 60f, 100f};
  private static final Long[] LONGS_1 = new Long[]{1L, 2L, 3L, 4L, 5L, null, 100L};
  private static final Long[] LONGS_2 = new Long[]{10L, 20L, 30L, 40L, 50L, 60L, 100L};
  private static final Double[] DOUBLES_1 = new Double[]{1.0, 10.0, 100.0, 5.3, 50.0, 100.0, null};
  private static final Double[] DOUBLES_2 = new Double[]{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 100.0};

  @Test
  public void testAdd() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector bcv1 = ColumnVector.fromBoxedBytes(BYTES_1);
         ColumnVector fcv1 = ColumnVector.fromBoxedFloats(FLOATS_1);
         ColumnVector fcv2 = ColumnVector.fromBoxedFloats(FLOATS_2);
         ColumnVector lcv1 = ColumnVector.fromBoxedLongs(LONGS_1);
         ColumnVector lcv2 = ColumnVector.fromBoxedLongs(LONGS_2);
         ColumnVector dcv1 = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector dcv2 = ColumnVector.fromBoxedDoubles(DOUBLES_2)) {
      try (ColumnVector add = icv1.add(icv2);
           ColumnVector expected = ColumnVector.fromBoxedInts(11, 22, 33, 44, 55, null, 200)) {
        assertColumnsAreEqual(expected, add, "int32");
      }

      try (ColumnVector add = icv1.add(bcv1);
           ColumnVector expected = ColumnVector.fromBoxedInts(0, 9, 126, null, 55, null, 200)) {
        assertColumnsAreEqual(expected, add, "int32 + byte");
      }

      try (ColumnVector add = fcv1.add(fcv2);
           ColumnVector expected = ColumnVector.fromBoxedFloats(11f, 30f, 130f, 45.3f, 100f, 160f
               , null)) {
        assertColumnsAreEqual(expected, add, "float32");
      }

      try (ColumnVector addIntFirst = icv1.add(fcv2, DType.FLOAT32);
           ColumnVector addFloatFirst = fcv2.add(icv1);) {
        assertColumnsAreEqual(addIntFirst, addFloatFirst, "int + float vs float + int");
      }

      try (ColumnVector add = lcv1.add(lcv2);
           ColumnVector expected = ColumnVector.fromBoxedLongs(11L, 22L, 33L, 44L, 55L, null,
               200L)) {
        assertColumnsAreEqual(expected, add, "int64");
      }

      try (ColumnVector add = lcv1.add(bcv1);
           ColumnVector expected = ColumnVector.fromBoxedLongs(0L, 9L, 126L, null, 55L, null,
               200L)) {
        assertColumnsAreEqual(expected, add, "int64 + byte");
      }

      try (ColumnVector add = dcv1.add(dcv2);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(11.0, 30.0, 130.0, 45.3, 100.0,
               160.0, null)) {
        assertColumnsAreEqual(expected, add, "float64");
      }

      try (ColumnVector addIntFirst = icv1.add(dcv2, DType.FLOAT64);
           ColumnVector addDoubleFirst = dcv2.add(icv1);) {
        assertColumnsAreEqual(addIntFirst, addDoubleFirst, "int + double vs double + int");
      }

      try (ColumnVector add = lcv1.add(Scalar.fromFloat(1.1f));
           ColumnVector expected = ColumnVector.fromBoxedFloats(2.1f, 3.1f, 4.1f, 5.1f, 6.1f,
               null, 101.1f)) {
        assertColumnsAreEqual(expected, add, "int64 + scalar float");
      }

      try (ColumnVector add = Scalar.fromShort((short) 100).add(bcv1);
           ColumnVector expected = ColumnVector.fromBoxedShorts((short) 99, (short) 107,
               (short) 223, null, (short) 150,
               (short) 160, (short) 200)) {
        assertColumnsAreEqual(expected, add, "scalar short + byte");
      }
    }
  }

  @Test
  public void testSub() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector bcv1 = ColumnVector.fromBoxedBytes(BYTES_1);
         ColumnVector fcv1 = ColumnVector.fromBoxedFloats(FLOATS_1);
         ColumnVector fcv2 = ColumnVector.fromBoxedFloats(FLOATS_2);
         ColumnVector lcv1 = ColumnVector.fromBoxedLongs(LONGS_1);
         ColumnVector lcv2 = ColumnVector.fromBoxedLongs(LONGS_2);
         ColumnVector dcv1 = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector dcv2 = ColumnVector.fromBoxedDoubles(DOUBLES_2)) {
      try (ColumnVector sub = icv1.sub(icv2);
           ColumnVector expected = ColumnVector.fromBoxedInts(-9, -18, -27, -36, -45, null, 0)) {
        assertColumnsAreEqual(expected, sub, "int32");
      }

      try (ColumnVector sub = icv1.sub(bcv1);
           ColumnVector expected = ColumnVector.fromBoxedInts(2, -5, -120, null, -45, null, 0)) {
        assertColumnsAreEqual(expected, sub, "int32 - byte");
      }

      try (ColumnVector sub = fcv1.sub(fcv2);
           ColumnVector expected = ColumnVector.fromBoxedFloats(-9f, -10f, 70f, -34.7f, 0f, 40f,
               null)) {
        assertColumnsAreEqual(expected, sub, "float32");
      }

      try (ColumnVector sub = icv1.sub(fcv2, DType.FLOAT32);
           ColumnVector expected = ColumnVector.fromBoxedFloats(-9f, -18f, -27f, -36f, -45f, null
               , 0f)) {
        assertColumnsAreEqual(expected, sub, "int - float");
      }

      try (ColumnVector sub = lcv1.sub(lcv2);
           ColumnVector expected = ColumnVector.fromBoxedLongs(-9L, -18L, -27L, -36L, -45L, null,
               0L)) {
        assertColumnsAreEqual(expected, sub, "int64");
      }

      try (ColumnVector sub = lcv1.sub(bcv1);
           ColumnVector expected = ColumnVector.fromBoxedLongs(2L, -5L, -120L, null, -45L, null,
               0L)) {
        assertColumnsAreEqual(expected, sub, "int64 - byte");
      }

      try (ColumnVector sub = dcv1.sub(dcv2);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(-9.0, -10.0, 70.0, -34.7, 0.0,
               40.0, null)) {
        assertColumnsAreEqual(expected, sub, "float64");
      }

      try (ColumnVector sub = dcv2.sub(icv1);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(9.0, 18.0, 27.0, 36.0, 45.0,
               null, 0.0)) {
        assertColumnsAreEqual(expected, sub, "double - int");
      }

      try (ColumnVector sub = lcv1.sub(Scalar.fromFloat(1.1f));
           ColumnVector expected = ColumnVector.fromBoxedFloats(-0.1f, 0.9f, 1.9f, 2.9f, 3.9f,
               null, 98.9f)) {
        assertColumnsAreEqual(expected, sub, "int64 - scalar float");
      }

      try (ColumnVector sub = Scalar.fromShort((short) 100).sub(bcv1);
           ColumnVector expected = ColumnVector.fromBoxedShorts((short) 101, (short) 93,
               (short) -23, null,
               (short) 50, (short) 40, (short) 0)) {
        assertColumnsAreEqual(expected, sub, "scalar short - byte");
      }
    }
  }

  // The rest of the tests are very basic to ensure that operations plumbing is in place, not to
  // exhaustively test
  // The underlying implementation.

  @Test
  public void testMul() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.mul(dcv);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(1 * 1.0, 2 * 10.0, 3 * 100.0,
               4 * 5.3, 5 * 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 * double");
      }

      try (ColumnVector answer = icv.mul(Scalar.fromFloat(1.1f));
           ColumnVector expected = ColumnVector.fromBoxedFloats(1 * 1.1f, 2 * 1.1f, 3 * 1.1f,
               4 * 1.1f, 5 * 1.1f, null, 100 * 1.1f)) {
        assertColumnsAreEqual(expected, answer, "int64 * scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).mul(icv);
           ColumnVector expected = ColumnVector.fromBoxedInts(100 * 1, 100 * 2, 100 * 3, 100 * 4,
               100 * 5, null, 100 * 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short * int32");
      }
    }
  }

  @Test
  public void testDiv() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.div(dcv);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(1 / 1.0, 2 / 10.0, 3 / 100.0,
               4 / 5.3, 5 / 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 / double");
      }

      try (ColumnVector answer = icv.div(Scalar.fromFloat(1.1f));
           ColumnVector expected = ColumnVector.fromBoxedFloats(1 / 1.1f, 2 / 1.1f, 3 / 1.1f,
               4 / 1.1f, 5 / 1.1f, null, 100 / 1.1f)) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).div(icv);
           ColumnVector expected = ColumnVector.fromBoxedInts(100 / 1, 100 / 2, 100 / 3, 100 / 4,
               100 / 5, null, 100 / 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short / int32");
      }
    }
  }

  @Test
  public void testTrueDiv() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.trueDiv(dcv);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(1 / 1.0, 2 / 10.0, 3 / 100.0,
               4 / 5.3, 5 / 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 / double");
      }

      try (ColumnVector answer = icv.trueDiv(Scalar.fromFloat(1.1f));
           ColumnVector expected = ColumnVector.fromBoxedFloats(1 / 1.1f, 2 / 1.1f, 3 / 1.1f,
               4 / 1.1f, 5 / 1.1f, null, 100 / 1.1f)) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).trueDiv(icv);
           ColumnVector expected = ColumnVector.fromBoxedInts(100 / 1, 100 / 2, 100 / 3, 100 / 4,
               100 / 5, null, 100 / 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short / int32");
      }
    }
  }

  @Test
  public void testFloorDiv() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.floorDiv(dcv);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.floor(1 / 1.0),
               Math.floor(2 / 10.0), Math.floor(3 / 100.0),
               Math.floor(4 / 5.3), Math.floor(5 / 50.0), null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 / double");
      }

      try (ColumnVector answer = icv.floorDiv(Scalar.fromFloat(1.1f));
           ColumnVector expected = ColumnVector.fromBoxedFloats((float) Math.floor(1 / 1.1f),
               (float) Math.floor(2 / 1.1f),
               (float) Math.floor(3 / 1.1f), (float) Math.floor(4 / 1.1f),
               (float) Math.floor(5 / 1.1f), null,
               (float) Math.floor(100 / 1.1f))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).floorDiv(icv);
           ColumnVector expected = ColumnVector.fromBoxedInts(100 / 1, 100 / 2, 100 / 3, 100 / 4,
               100 / 5, null, 100 / 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short / int32");
      }
    }
  }

  @Test
  public void testMod() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.mod(dcv);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(1 % 1.0, 2 % 10.0, 3 % 100.0,
               4 % 5.3, 5 % 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 % double");
      }

      try (ColumnVector answer = icv.mod(Scalar.fromFloat(1.1f));
           ColumnVector expected = ColumnVector.fromBoxedFloats(1 % 1.1f, 2 % 1.1f, 3 % 1.1f,
               4 % 1.1f, 5 % 1.1f, null, 100 % 1.1f)) {
        assertColumnsAreEqual(expected, answer, "int64 % scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).mod(icv);
           ColumnVector expected = ColumnVector.fromBoxedInts(100 % 1, 100 % 2, 100 % 3, 100 % 4,
               100 % 5, null, 100 % 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short % int32");
      }
    }
  }

  @Test
  public void testPow() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.pow(dcv);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(Math.pow(1, 1.0), Math.pow(2,
               10.0), Math.pow(3, 100.0),
               Math.pow(4, 5.3), Math.pow(5, 50.0), null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 pow double");
      }

      try (ColumnVector answer = icv.pow(Scalar.fromFloat(1.1f));
           ColumnVector expected = ColumnVector.fromBoxedFloats((float) Math.pow(1, 1.1f),
               (float) Math.pow(2, 1.1f),
               (float) Math.pow(3, 1.1f), (float) Math.pow(4, 1.1f), (float) Math.pow(5, 1.1f),
               null,
               (float) Math.pow(100, 1.1f))) {
        assertColumnsAreEqual(expected, answer, "int64 pow scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).pow(icv);
           ColumnVector expected = ColumnVector.fromBoxedInts((int) Math.pow(100, 1),
               (int) Math.pow(100, 2),
               (int) Math.pow(100, 3), (int) Math.pow(100, 4), (int) Math.pow(100, 5), null,
               (int) Math.pow(100, 100))) {
        assertColumnsAreEqual(expected, answer, "scalar short pow int32");
      }
    }
  }

  @Test
  public void testEqual() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.equalTo(dcv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 == 1.0, 2 == 10.0, 3 == 100.0
               , 4 == 5.3, 5 == 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 == double");
      }

      try (ColumnVector answer = icv.equalTo(Scalar.fromFloat(1.0f));
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 == 1.0f, 2 == 1.0f, 3 == 1.0f
               , 4 == 1.0f, 5 == 1.0f, null, 100 == 1.0f)) {
        assertColumnsAreEqual(expected, answer, "int64 == scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).equalTo(icv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(100 == 1, 100 == 2, 100 == 3,
               100 == 4, 100 == 5, null, 100 == 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short == int32");
      }
    }
  }

  @Test
  public void testStringCategoryEqualScalar() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("b");

      try (ColumnVector answer = a.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testNotEqual() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.notEqualTo(dcv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 != 1.0, 2 != 10.0, 3 != 100.0
               , 4 != 5.3,
               5 != 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 != double");
      }

      try (ColumnVector answer = icv.notEqualTo(Scalar.fromFloat(1.0f));
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 != 1.0f, 2 != 1.0f, 3 != 1.0f
               , 4 != 1.0f,
               5 != 1.0f, null, 100 != 1.0f)) {
        assertColumnsAreEqual(expected, answer, "int64 != scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).notEqualTo(icv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(100 != 1, 100 != 2, 100 != 3,
               100 != 4,
               100 != 5, null, 100 != 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short != int32");
      }
    }
  }

  @Test
  public void testStringCategoryNotEqualScalar() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("b");

      try (ColumnVector answer = a.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testLessThan() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.lessThan(dcv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 < 1.0, 2 < 10.0, 3 < 100.0,
               4 < 5.3, 5 < 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 < double");
      }

      try (ColumnVector answer = icv.lessThan(Scalar.fromFloat(1.0f));
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 < 1.0f, 2 < 1.0f, 3 < 1.0f,
               4 < 1.0f, 5 < 1.0f, null, 100 < 1.0f)) {
        assertColumnsAreEqual(expected, answer, "int64 < scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).lessThan(icv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(100 < 1, 100 < 2, 100 < 3,
               100 < 4, 100 < 5, null, 100 < 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short < int32");
      }
    }
  }


  @Test
  public void testStringCategoryLessThanScalar() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("b");

      try (ColumnVector answer = a.lessThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.lessThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.lessThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testGreaterThan() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.greaterThan(dcv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 > 1.0, 2 > 10.0, 3 > 100.0,
               4 > 5.3, 5 > 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 > double");
      }

      try (ColumnVector answer = icv.greaterThan(Scalar.fromFloat(1.0f));
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 > 1.0f, 2 > 1.0f, 3 > 1.0f,
               4 > 1.0f, 5 > 1.0f, null, 100 > 1.0f)) {
        assertColumnsAreEqual(expected, answer, "int64 > scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).greaterThan(icv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(100 > 1, 100 > 2, 100 > 3,
               100 > 4, 100 > 5, null, 100 > 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short > int32");
      }
    }
  }

  @Test
  public void testStringCategoryGreaterThanScalar() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("b");

      try (ColumnVector answer = a.greaterThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.greaterThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.greaterThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testLessOrEqualTo() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.lessOrEqualTo(dcv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 <= 1.0, 2 <= 10.0, 3 <= 100.0
               , 4 <= 5.3, 5 <= 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 <= double");
      }

      try (ColumnVector answer = icv.lessOrEqualTo(Scalar.fromFloat(1.0f));
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 <= 1.0f, 2 <= 1.0f, 3 <= 1.0f
               , 4 <= 1.0f, 5 <= 1.0f, null, 100 <= 1.0f)) {
        assertColumnsAreEqual(expected, answer, "int64 <= scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).lessOrEqualTo(icv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(100 <= 1, 100 <= 2, 100 <= 3,
               100 <= 4, 100 <= 5, null, 100 <= 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short <= int32");
      }
    }
  }

  @Test
  public void testStringCategoryLessOrEqualToScalar() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("b");

      try (ColumnVector answer = a.lessOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.lessOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.lessOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testGreaterOrEqualTo() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.greaterOrEqualTo(dcv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 >= 1.0, 2 >= 10.0, 3 >= 100.0
               , 4 >= 5.3, 5 >= 50.0, null, null)) {
        assertColumnsAreEqual(expected, answer, "int32 >= double");
      }

      try (ColumnVector answer = icv.greaterOrEqualTo(Scalar.fromFloat(1.0f));
           ColumnVector expected = ColumnVector.fromBoxedBooleans(1 >= 1.0f, 2 >= 1.0f, 3 >= 1.0f
               , 4 >= 1.0f, 5 >= 1.0f, null, 100 >= 1.0f)) {
        assertColumnsAreEqual(expected, answer, "int64 >= scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).greaterOrEqualTo(icv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(100 >= 1, 100 >= 2, 100 >= 3,
               100 >= 4, 100 >= 5, null, 100 >= 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short >= int32");
      }
    }
  }

  @Test
  public void testStringCategoryGreaterOrEqualToScalar() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("b");

      try (ColumnVector answer = a.greaterOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.greaterOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.greaterOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  // TODO do we want assertions for non int-types?
  @Test
  public void testBitAnd() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2)) {
      try (ColumnVector answer = icv1.bitAnd(icv2);
           ColumnVector expected = ColumnVector.fromBoxedInts(1 & 10, 2 & 20, 3 & 30, 4 & 40,
               5 & 50, null, 100 & 100)) {
        assertColumnsAreEqual(expected, answer, "int32 & int32");
      }

      try (ColumnVector answer = icv1.bitAnd(Scalar.fromInt(0x01));
           ColumnVector expected = ColumnVector.fromBoxedInts(1 & 0x01, 2 & 0x01, 3 & 0x01,
               4 & 0x01, 5 & 0x01, null, 100 & 0x01)) {
        assertColumnsAreEqual(expected, answer, "int32 & scalar int32");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).bitAnd(icv1);
           ColumnVector expected = ColumnVector.fromBoxedInts(1 & 100, 2 & 100, 3 & 100, 4 & 100,
               5 & 100, null, 100 & 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short & int32");
      }
    }
  }

  @Test
  public void testBitOr() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2)) {
      try (ColumnVector answer = icv1.bitOr(icv2);
           ColumnVector expected = ColumnVector.fromBoxedInts(1 | 10, 2 | 20, 3 | 30, 4 | 40,
               5 | 50, null, 100 | 100)) {
        assertColumnsAreEqual(expected, answer, "int32 | int32");
      }

      try (ColumnVector answer = icv1.bitOr(Scalar.fromInt(0x01));
           ColumnVector expected = ColumnVector.fromBoxedInts(1 | 0x01, 2 | 0x01, 3 | 0x01,
               4 | 0x01, 5 | 0x01, null, 100 | 0x01)) {
        assertColumnsAreEqual(expected, answer, "int32 | scalar int32");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).bitOr(icv1);
           ColumnVector expected = ColumnVector.fromBoxedInts(1 | 100, 2 | 100, 3 | 100, 4 | 100,
               5 | 100, null, 100 | 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short | int32");
      }
    }
  }

  @Test
  public void testBitXor() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2)) {
      try (ColumnVector answer = icv1.bitXor(icv2);
           ColumnVector expected = ColumnVector.fromBoxedInts(1 ^ 10, 2 ^ 20, 3 ^ 30, 4 ^ 40,
               5 ^ 50, null, 100 ^ 100)) {
        assertColumnsAreEqual(expected, answer, "int32 ^ int32");
      }

      try (ColumnVector answer = icv1.bitXor(Scalar.fromInt(0x01));
           ColumnVector expected = ColumnVector.fromBoxedInts(1 ^ 0x01, 2 ^ 0x01, 3 ^ 0x01,
               4 ^ 0x01, 5 ^ 0x01, null, 100 ^ 0x01)) {
        assertColumnsAreEqual(expected, answer, "int32 ^ scalar int32");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).bitXor(icv1);
           ColumnVector expected = ColumnVector.fromBoxedInts(100 ^ 1, 100 ^ 2, 100 ^ 3, 100 ^ 4,
               100 ^ 5, null, 100 ^ 100)) {
        assertColumnsAreEqual(expected, answer, "scalar short ^ int32");
      }
    }
  }
}