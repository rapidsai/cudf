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
import org.opentest4j.AssertionFailedError;

import java.util.Arrays;
import java.util.stream.IntStream;

import static ai.rapids.cudf.QuantileMethod.*;
import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class ColumnVectorTest extends CudfTestBase {

  public static final double DELTA = 0.0001;

  // c = a * a - a
  static String ptx = "***(" +
      "      .func _Z1fPii(" +
      "        .param .b64 _Z1fPii_param_0," +
      "        .param .b32 _Z1fPii_param_1" +
      "  )" +
      "  {" +
      "        .reg .b32       %r<4>;" +
      "        .reg .b64       %rd<3>;" +
      "    ld.param.u64    %rd1, [_Z1fPii_param_0];" +
      "    ld.param.u32    %r1, [_Z1fPii_param_1];" +
      "    cvta.to.global.u64      %rd2, %rd1;" +
      "    mul.lo.s32      %r2, %r1, %r1;" +
      "    sub.s32         %r3, %r2, %r1;" +
      "    st.global.u32   [%rd2], %r3;" +
      "    ret;" +
      "  }" +
      ")***";

  @Test
  void testTransformVector() {
    try (ColumnVector cv = ColumnVector.fromBoxedInts(2,3,null,4);
         ColumnVector cv1 = cv.transform(ptx, true);
         ColumnVector expected = ColumnVector.fromBoxedInts(2*2-2, 3*3-3, null, 4*4-4)) {
      for (int i = 0 ; i < cv1.getRowCount() ; i++) {
        cv1.ensureOnHost();
        assertEquals(expected.isNull(i), cv1.isNull(i));
        if (!expected.isNull(i)) {
          assertEquals(expected.getInt(i), cv1.getInt(i));
        }
      }
    }
  }

  @Test
  void testStringCreation() {
    try (ColumnVector cv = ColumnVector.fromStrings("d", "sd", "sde", null, "END");
         ColumnVector orig = ColumnVector.fromStrings("d", "sd", "sde", null, "END")) {
      TableTest.assertColumnsAreEqual(orig, cv);
      cv.dropHostData();
      cv.ensureOnHost();
      TableTest.assertColumnsAreEqual(orig, cv);
    }
  }

  @Test
  void testDataMovement() {
    try (ColumnVector vec = ColumnVector.fromBoxedInts(1, 2, 3, 4, null, 6)) {
      assert vec.hasDeviceData();
      assert vec.hasHostData();
      vec.dropHostData();
      assert !vec.hasHostData();
      assert vec.hasDeviceData();
      vec.dropDeviceData();
      assert vec.hasHostData();
      assert !vec.hasDeviceData();
    }
  }

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
  void testConcatStrings() {
    try (ColumnVector v0 = ColumnVector.fromStrings("0","1","2",null);
         ColumnVector v1 = ColumnVector.fromStrings(null, "5", "6","7");
         ColumnVector expected = ColumnVector.fromStrings(
           "0","1","2",null,
           null,"5","6","7");
         ColumnVector v = ColumnVector.concatenate(v0, v1)) {
      assertColumnsAreEqual(v, expected);
    }
  }

  @Test
  void testConcatTimestamps() {
    try (ColumnVector v0 = ColumnVector.timestampMicroSecondsFromBoxedLongs(0L, 1L, 2L, null);
         ColumnVector v1 = ColumnVector.timestampMicroSecondsFromBoxedLongs(null, 5L, 6L, 7L);
         ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
           0L, 1L, 2L, null,
           null, 5L, 6L, 7L);
         ColumnVector v = ColumnVector.concatenate(v0, v1)) {
      assertColumnsAreEqual(v, expected);
    }
  }

  @Test
  void isNotNullTestEmptyColumn() {
    try (ColumnVector v = ColumnVector.fromBoxedInts();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void isNotNullTest() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(1, 2, null, 4, null, 6);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, false, true, false, true);
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void isNotNullTestAllNulls() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(null, null, null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false, false, false);
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void isNotNullTestAllNotNulls() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(1,2,3,4,5,6);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true, true, true);
         ColumnVector result = v.isNotNull()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void isNullTest() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(1, 2, null, 4, null, 6);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, false, true, false);
         ColumnVector result = v.isNull()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void isNullTestEmptyColumn() {
    try (ColumnVector v = ColumnVector.fromBoxedInts();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         ColumnVector result = v.isNull()) {
      assertColumnsAreEqual(expected, result);
    }
  }

   @Test
  void isNanTestWithNulls() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(null, null, Double.NaN, null, Double.NaN, null);
         ColumnVector vF = ColumnVector.fromBoxedFloats(null, null, Float.NaN, null, Float.NaN, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, false, true, false);
         ColumnVector result = v.isNan();
         ColumnVector resultF = vF.isNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void isNanForTypeMismatch() {
    assertThrows(CudfException.class, () -> {
      try (ColumnVector v = ColumnVector.fromStrings("foo", "bar", "baz");
           ColumnVector result = v.isNan()) {}
    });
  }

  @Test
  void isNanTest() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(1.0, 2.0, Double.NaN, 4.0, Double.NaN, 6.0);
         ColumnVector vF = ColumnVector.fromBoxedFloats(1.1f, 2.2f, Float.NaN, 4.4f, Float.NaN, 6.6f);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, false, true, false);
         ColumnVector result = v.isNan();
         ColumnVector resultF = vF.isNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void isNanTestEmptyColumn() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles();
         ColumnVector vF = ColumnVector.fromBoxedFloats();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         ColumnVector result = v.isNan();
         ColumnVector resultF = vF.isNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void isNanTestAllNotNans() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
         ColumnVector vF = ColumnVector.fromBoxedFloats(1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false, false, false);
         ColumnVector result = v.isNan();
         ColumnVector resultF = vF.isNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void isNanTestAllNans() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN);
         ColumnVector vF = ColumnVector.fromBoxedFloats(Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true, true, true);
         ColumnVector result = v.isNan();
         ColumnVector resultF = vF.isNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void isNotNanTestWithNulls() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(null, null, Double.NaN, null, Double.NaN, null);
         ColumnVector vF = ColumnVector.fromBoxedFloats(null, null, Float.NaN, null, Float.NaN, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, false, true, false, true);
         ColumnVector result = v.isNotNan();
         ColumnVector resultF = vF.isNotNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void isNotNanForTypeMismatch() {
    assertThrows(CudfException.class, () -> {
      try (ColumnVector v = ColumnVector.fromStrings("foo", "bar", "baz");
           ColumnVector result = v.isNotNan()) {}
    });
  }

  @Test
  void isNotNanTest() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(1.0, 2.0, Double.NaN, 4.0, Double.NaN, 6.0);
         ColumnVector vF = ColumnVector.fromBoxedFloats(1.1f, 2.2f, Float.NaN, 4.4f, Float.NaN, 6.6f);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, false, true, false, true);
         ColumnVector result = v.isNotNan();
         ColumnVector resultF = vF.isNotNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void isNotNanTestEmptyColumn() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles();
         ColumnVector vF = ColumnVector.fromBoxedFloats();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         ColumnVector result = v.isNotNan();
         ColumnVector resultF = vF.isNotNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void isNotNanTestAllNotNans() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
         ColumnVector vF = ColumnVector.fromBoxedFloats(1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true, true, true);
         ColumnVector result = v.isNotNan();
         ColumnVector resultF = vF.isNotNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void isNotNanTestAllNans() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN);
         ColumnVector vF = ColumnVector.fromBoxedFloats(Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false, false, false);
         ColumnVector result = v.isNotNan();
         ColumnVector resultF = vF.isNotNan()) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expected, resultF);
    }
  }

  @Test
  void testGetDeviceMemorySizeNonStrings() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(1, 2, 3, 4, 5, 6);
         ColumnVector v1 = ColumnVector.fromBoxedInts(1, 2, 3, null, null, 4, 5, 6)) {
      assertEquals(24, v0.getDeviceMemorySize()); // (6*4B)
      assertEquals(96, v1.getDeviceMemorySize()); // (8*4B) + 64B(for validity vector)
    }
  }

  @Test
  void testGetDeviceMemorySizeStrings() {
    try (ColumnVector v0 = ColumnVector.fromStrings("onetwothree", "four", "five");
         ColumnVector v1 = ColumnVector.fromStrings("onetwothree", "four", null, "five")) {
      assertEquals(35, v0.getDeviceMemorySize()); //19B data + 4*4B offsets = 35
      assertEquals(103, v1.getDeviceMemorySize()); //19B data + 5*4B + 64B validity vector = 103B
    }
  }

  @Test
  void testFromScalarZeroRows() {
    for (DType type : DType.values()) {
      Scalar s = null;
      try {
        switch (type) {
        case BOOL8:
          s = Scalar.fromBool(true);
          break;
        case INT8:
          s = Scalar.fromByte((byte) 5);
          break;
        case INT16:
          s = Scalar.fromShort((short) 12345);
          break;
        case INT32:
          s = Scalar.fromInt(123456789);
          break;
        case INT64:
          s = Scalar.fromLong(1234567890123456789L);
          break;
        case FLOAT32:
          s = Scalar.fromFloat(1.2345f);
          break;
        case FLOAT64:
          s = Scalar.fromDouble(1.23456789);
          break;
        case TIMESTAMP_DAYS:
          s = Scalar.timestampDaysFromInt(12345);
          break;
        case TIMESTAMP_SECONDS:
        case TIMESTAMP_MILLISECONDS:
        case TIMESTAMP_MICROSECONDS:
        case TIMESTAMP_NANOSECONDS:
          s = Scalar.timestampFromLong(type, 1234567890123456789L);
          break;
        case STRING:
          s = Scalar.fromString("hello, world!");
          break;
        case EMPTY:
          continue;
        default:
          throw new IllegalArgumentException("Unexpected type: " + type);
        }

        try (ColumnVector c = ColumnVector.fromScalar(s, 0)) {
          assertEquals(type, c.getType());
          assertEquals(0, c.getRowCount());
          assertEquals(0, c.getNullCount());
        }
      } finally {
        if (s != null) {
          s.close();
        }
      }
    }
  }

  @Test
  void testFromScalar() {
    final int rowCount = 4;
    for (DType type : DType.values()) {
      Scalar s = null;
      ColumnVector expected = null;
      ColumnVector result = null;
      try {
        switch (type) {
        case BOOL8:
          s = Scalar.fromBool(true);
          expected = ColumnVector.fromBoxedBooleans(true, true, true, true);
          break;
        case INT8: {
          byte v = (byte) 5;
          s = Scalar.fromByte(v);
          expected = ColumnVector.fromBoxedBytes(v, v, v, v);
          break;
        }
        case INT16: {
          short v = (short) 12345;
          s = Scalar.fromShort(v);
          expected = ColumnVector.fromBoxedShorts((short) 12345, (short) 12345, (short) 12345, (short) 12345);
          break;
        }
        case INT32: {
          int v = 123456789;
          s = Scalar.fromInt(v);
          expected = ColumnVector.fromBoxedInts(v, v, v, v);
          break;
        }
        case INT64: {
          long v = 1234567890123456789L;
          s = Scalar.fromLong(v);
          expected = ColumnVector.fromBoxedLongs(v, v, v, v);
          break;
        }
        case FLOAT32: {
          float v = 1.2345f;
          s = Scalar.fromFloat(v);
          expected = ColumnVector.fromBoxedFloats(v, v, v, v);
          break;
        }
        case FLOAT64: {
          double v = 1.23456789;
          s = Scalar.fromDouble(v);
          expected = ColumnVector.fromBoxedDoubles(v, v, v, v);
          break;
        }
        case TIMESTAMP_DAYS: {
          int v = 12345;
          s = Scalar.timestampDaysFromInt(v);
          expected = ColumnVector.daysFromInts(v, v, v, v);
          break;
        }
        case TIMESTAMP_SECONDS: {
          long v = 1234567890123456789L;
          s = Scalar.timestampFromLong(type, v);
          expected = ColumnVector.timestampSecondsFromLongs(v, v, v, v);
          break;
        }
        case TIMESTAMP_MILLISECONDS: {
          long v = 1234567890123456789L;
          s = Scalar.timestampFromLong(type, v);
          expected = ColumnVector.timestampMilliSecondsFromLongs(v, v, v, v);
          break;
        }
        case TIMESTAMP_MICROSECONDS: {
          long v = 1234567890123456789L;
          s = Scalar.timestampFromLong(type, v);
          expected = ColumnVector.timestampMicroSecondsFromLongs(v, v, v, v);
          break;
        }
        case TIMESTAMP_NANOSECONDS: {
          long v = 1234567890123456789L;
          s = Scalar.timestampFromLong(type, v);
          expected = ColumnVector.timestampNanoSecondsFromLongs(v, v, v, v);
          break;
        }
        case STRING: {
          String v = "hello, world!";
          s = Scalar.fromString(v);
          expected = ColumnVector.fromStrings(v, v, v, v);
          break;
        }
        case EMPTY:
          continue;
        default:
          throw new IllegalArgumentException("Unexpected type: " + type);
        }

        result = ColumnVector.fromScalar(s, rowCount);
        assertColumnsAreEqual(expected, result);
      } finally {
        if (s != null) {
          s.close();
        }
        if (expected != null) {
          expected.close();
        }
        if (result != null) {
          result.close();
        }
      }
    }
  }

  @Test
  void testFromScalarNull() {
    final int rowCount = 4;
    for (DType type : DType.values()) {
      if (type == DType.EMPTY) {
        continue;
      }
      try (Scalar s = Scalar.fromNull(type);
           ColumnVector c = ColumnVector.fromScalar(s, rowCount)) {
        assertEquals(type, c.getType());
        assertEquals(rowCount, c.getRowCount());
        assertEquals(rowCount, c.getNullCount());
        c.ensureOnHost();
        for (int i = 0; i < rowCount; ++i) {
          assertTrue(c.isNull(i));
        }
      }
    }
  }

  @Test
  void testFromScalarNullByte() {
    int numNulls = 3000;
    try (Scalar s = Scalar.fromNull(DType.INT8);
         ColumnVector input = ColumnVector.fromScalar(s, numNulls)) {
      assertEquals(numNulls, input.getRowCount());
      assertEquals(input.getNullCount(), numNulls);
      input.ensureOnHost();
      for (int i = 0; i < numNulls; i++){
        assertTrue(input.isNull(i));
      }
    }
  }

  @Test
  void testReplaceEmptyColumn() {
    try (ColumnVector input = ColumnVector.fromBoxedBooleans();
         ColumnVector expected = ColumnVector.fromBoxedBooleans();
         Scalar s = Scalar.fromBool(false);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullBoolsWithAllNulls() {
    try (ColumnVector input = ColumnVector.fromBoxedBooleans(null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false);
         Scalar s = Scalar.fromBool(false);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceSomeNullBools() {
    try (ColumnVector input = ColumnVector.fromBoxedBooleans(false, null, null, false);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false);
         Scalar s = Scalar.fromBool(true);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullIntegersWithAllNulls() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(null, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedInts(0, 0, 0, 0);
         Scalar s = Scalar.fromInt(0);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceSomeNullIntegers() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null);
         ColumnVector expected = ColumnVector.fromBoxedInts(1, 2, 999, 4, 999);
         Scalar s = Scalar.fromInt(999);
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(result, expected);
    }
  }

  @Test
  void testReplaceNullsFailsOnTypeMismatch() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null);
         Scalar s = Scalar.fromBool(true)) {
      assertThrows(CudfException.class, () -> input.replaceNulls(s).close());
    }
  }

  @Test
  void testReplaceNullsWithNullScalar() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(1, 2, null, 4, null);
         Scalar s = Scalar.fromNull(input.getType());
         ColumnVector result = input.replaceNulls(s)) {
      assertColumnsAreEqual(input, result);
    }
  }

  static QuantileMethod[] methods = {LINEAR, LOWER, HIGHER, MIDPOINT, NEAREST};
  static double[] quantiles = {0.0, 0.25, 0.33, 0.5, 1.0};

  @Test
  void testQuantilesOnIntegerInput() {
    double[][] exactExpected = {
        {-1.0,   1.0,   1.0,   2.5,   9.0},  // LINEAR
        {  -1,     1,     1,     2,     9},  // LOWER
        {  -1,     1,     1,     3,     9},  // HIGHER
        {-1.0,   1.0,   1.0,   2.5,   9.0},  // MIDPOINT
        {  -1,     1,     1,     2,     9}}; // NEAREST

    try (ColumnVector cv = ColumnVector.fromBoxedInts(7, 0, 3, 4, 2, 1, -1, 1, 6, 9)) {
      // sorted: -1, 0, 1, 1, 2, 3, 4, 6, 7, 9
      for (int j = 0 ; j < quantiles.length ; j++) {
        for (int i = 0 ; i < methods.length ; i++) {
          try(Scalar result = cv.quantile(methods[i], quantiles[j])) {
            assertEquals(exactExpected[i][j], result.getDouble(), DELTA);
          }
        }
      }
    }
  }

  @Test
  void testQuantilesOnDoubleInput() {
    double[][] exactExpected = {
        {-1.01, 0.8, 0.9984, 2.13, 6.8},  // LINEAR
        {-1.01, 0.8,    0.8, 2.13, 6.8},  // LOWER
        {-1.01, 0.8,   1.11, 2.13, 6.8},  // HIGHER
        {-1.01, 0.8,  0.955, 2.13, 6.8},  // MIDPOINT
        {-1.01, 0.8,   1.11, 2.13, 6.8}}; // NEAREST

    try (ColumnVector cv = ColumnVector.fromBoxedDoubles(6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7)) {
      // sorted: -1.01, 0.15, 0.8, 1.11, 2.13, 3.4, 4.17, 5.7, 6.8
      for (int j = 0; j < quantiles.length ; j++) {
        for (int i = 0 ; i < methods.length ; i++) {
          try (Scalar result = cv.quantile(methods[i], quantiles[j])) {
            assertEquals(exactExpected[i][j], result.getDouble(), DELTA);
          }
        }
      }
    }
  }

  @Test
  void testSubvector() {
    try (ColumnVector vec = ColumnVector.fromBoxedInts(1, 2, 3, null, 5);
         ColumnVector expected = ColumnVector.fromBoxedInts(2, 3, null, 5);
         ColumnVector found = vec.subVector(1, 5)) {
      TableTest.assertColumnsAreEqual(expected, found);
    }

    try (ColumnVector vec = ColumnVector.fromStrings("1", "2", "3", null, "5");
         ColumnVector expected = ColumnVector.fromStrings("2", "3", null, "5");
         ColumnVector found = vec.subVector(1, 5)) {
      TableTest.assertColumnsAreEqual(expected, found);
    }
  }

  @Test
  void testSlice() {
    try(ColumnVector cv = ColumnVector.fromBoxedInts(10, 12, null, null, 18, 20, 22, 24, 26, 28)) {
      Integer[][] expectedSlice = {
          {12, null},
          {20, 22, 24, 26},
          {null, null},
          {}};

      ColumnVector[] slices = cv.slice(1, 3, 5, 9, 2, 4, 8, 8);

      try {
        for (int i = 0; i < slices.length; i++) {
          final int sliceIndex = i;
          ColumnVector slice = slices[sliceIndex];
          slice.ensureOnHost();
          assertEquals(expectedSlice[sliceIndex].length, slices[sliceIndex].getRowCount());
          IntStream.range(0, expectedSlice[sliceIndex].length).forEach(rowCount -> {
            if (expectedSlice[sliceIndex][rowCount] == null) {
              assertTrue(slices[sliceIndex].isNull(rowCount));
            } else {
              assertEquals(expectedSlice[sliceIndex][rowCount],
                  slices[sliceIndex].getInt(rowCount));
            }
          });
        }
        assertEquals(4, slices.length);
      } finally {
        for (int i = 0 ; i < slices.length ; i++) {
          if (slices[i] != null) {
            slices[i].close();
          }
        }
      }
    }
  }

  @Test
  void testStringSlice() {
    try(ColumnVector cv = ColumnVector.fromStrings("foo", "bar", null, null, "baz", "hello", "world", "cuda", "is", "great")) {
      String[][] expectedSlice = {
          {"foo", "bar"},
          {null, null, "baz"},
          {null, "baz", "hello"}};

      ColumnVector[] slices = cv.slice(0, 2, 2, 5, 3, 6);

      try {
        for (int i = 0; i < slices.length; i++) {
          final int sliceIndex = i;
          ColumnVector slice = slices[sliceIndex];
          slice.ensureOnHost();
          assertEquals(expectedSlice[sliceIndex].length, slices[sliceIndex].getRowCount());
          IntStream.range(0, expectedSlice[sliceIndex].length).forEach(rowCount -> {
            if (expectedSlice[sliceIndex][rowCount] == null) {
              assertTrue(slices[sliceIndex].isNull(rowCount));
            } else {
              assertEquals(expectedSlice[sliceIndex][rowCount],
                  slices[sliceIndex].getJavaString(rowCount));
            }
          });
        }
        assertEquals(3, slices.length);
      } finally {
        for (int i = 0 ; i < slices.length ; i++) {
          if (slices[i] != null) {
            slices[i].close();
          }
        }
      }
    }
  }

  @Test
  void testSplitWithArray() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try(ColumnVector cv = ColumnVector.fromBoxedInts(10, 12, null, null, 18, 20, 22, 24, 26, 28)) {
      Integer[][] expectedData = {
          {10},
          {12, null},
          {null, 18},
          {20, 22, 24, 26},
          {28}};

      ColumnVector[] splits = cv.split(1, 3, 5, 9);
      try {
        assertEquals(expectedData.length, splits.length);
        for (int splitIndex = 0; splitIndex < splits.length; splitIndex++) {
          ColumnVector subVec = splits[splitIndex];
          subVec.ensureOnHost();
          assertEquals(expectedData[splitIndex].length, subVec.getRowCount());
          for (int subIndex = 0; subIndex < expectedData[splitIndex].length; subIndex++) {
            Integer expected = expectedData[splitIndex][subIndex];
            if (expected == null) {
              assertTrue(subVec.isNull(subIndex));
            } else {
              assertEquals(expected, subVec.getInt(subIndex));
            }
          }
        }
      } finally {
        for (int i = 0 ; i < splits.length ; i++) {
          if (splits[i] != null) {
            splits[i].close();
          }
        }
      }
    }
  }

  @Test
  void testWithOddSlices() {
    try (ColumnVector cv = ColumnVector.fromBoxedInts(10, 12, null, null, 18, 20, 22, 24, 26, 28)) {
      assertThrows(CudfException.class, () -> cv.slice(1, 3, 5, 9, 2, 4, 8));
    }
  }

  @Test
  void testAppendStrings() {
    try (ColumnVector cv = ColumnVector.build(DType.STRING, 10, 0, (b) -> {
      b.append("123456789");
      b.append("1011121314151617181920");
      b.append("");
      b.appendNull();
    })) {
      assertEquals(4, cv.getRowCount());
      assertEquals("123456789", cv.getJavaString(0));
      assertEquals("1011121314151617181920", cv.getJavaString(1));
      assertEquals("", cv.getJavaString(2));
      assertTrue(cv.isNull(3));
    }
  }

  @Test
  void testStringLengths() {
    try (ColumnVector cv = ColumnVector.fromStrings("1", "12", null, "123", "1234");
      ColumnVector lengths = cv.getLengths()) {
      lengths.ensureOnHost();
      assertEquals(5, lengths.getRowCount());
      for (int i = 0 ; i < lengths.getRowCount() ; i++) {
        if (cv.isNull(i)) {
          assertTrue(lengths.isNull(i));
        } else {
          assertEquals(cv.getJavaString(i).length(), lengths.getInt(i));
        }
      }
    }
  }

  @Test
  void testGetByteCount() {
    try (ColumnVector cv = ColumnVector.fromStrings("1", "12", "123", null, "1234");
         ColumnVector byteLengthVector = cv.getByteCount()) {
      byteLengthVector.ensureOnHost();
      assertEquals(5, byteLengthVector.getRowCount());
      for (int i = 0; i < byteLengthVector.getRowCount(); i++) {
        if (cv.isNull(i)) {
          assertTrue(byteLengthVector.isNull(i));
        } else {
          assertEquals(cv.getJavaString(i).length(), byteLengthVector.getInt(i));

        }
      }
    }
  }

  @Test
  void testEmptyStringColumnOpts() {
    try (ColumnVector cv = ColumnVector.fromStrings()) {
      try (ColumnVector len = cv.getLengths()) {
        assertEquals(0, len.getRowCount());
      }

      try (ColumnVector mask = ColumnVector.fromBoxedBooleans();
           Table input = new Table(cv);
           Table filtered = input.filter(mask)) {
        assertEquals(0, filtered.getColumn(0).getRowCount());
      }

      try (ColumnVector len = cv.getByteCount()) {
        assertEquals(0, len.getRowCount());
      }

      try (ColumnVector lower = cv.lower();
           ColumnVector upper = cv.upper()) {
        assertColumnsAreEqual(cv, lower);
        assertColumnsAreEqual(cv, upper);
      }
    }
  }

  @Test
  void testStringManipulation() {
    try (ColumnVector v = ColumnVector.fromStrings("a", "B", "cd", "\u0480\u0481", "E\tf",
                                                   "g\nH", "IJ\"\u0100\u0101\u0500\u0501",
                                                   "kl m", "Nop1", "\\qRs2", "3tuV\'",
                                                   "wX4Yz", "\ud720\ud721");
         ColumnVector e_lower = ColumnVector.fromStrings("a", "b", "cd", "\u0481\u0481", "e\tf",
                                                         "g\nh", "ij\"\u0101\u0101\u0501\u0501",
                                                         "kl m", "nop1", "\\qrs2", "3tuv\'",
                                                         "wx4yz", "\ud720\ud721");
         ColumnVector e_upper = ColumnVector.fromStrings("A", "B", "CD", "\u0480\u0480", "E\tF",
                                                         "G\nH", "IJ\"\u0100\u0100\u0500\u0500",
                                                         "KL M", "NOP1", "\\QRS2", "3TUV\'",
                                                         "WX4YZ", "\ud720\ud721");
         ColumnVector lower = v.lower();
         ColumnVector upper = v.upper()) {
      assertColumnsAreEqual(lower, e_lower);
      assertColumnsAreEqual(upper, e_upper);
    }
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4);
           ColumnVector lower = cv.lower()) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4);
           ColumnVector upper = cv.upper()) {}
    });
  }

  @Test
  void testStringManipulationWithNulls() {
    // Special characters in order of usage, capital and small cyrillic koppa
    // Latin A with macron, and cyrillic komi de
    // \ud720 and \ud721 are UTF-8 characters without corresponding upper and lower characters
    try (ColumnVector v = ColumnVector.fromStrings("a", "B", "cd", "\u0480\u0481", "E\tf",
                                                   "g\nH", "IJ\"\u0100\u0101\u0500\u0501",
                                                   "kl m", "Nop1", "\\qRs2", null,
                                                   "3tuV\'", "wX4Yz", "\ud720\ud721");
         ColumnVector e_lower = ColumnVector.fromStrings("a", "b", "cd", "\u0481\u0481", "e\tf",
                                                         "g\nh", "ij\"\u0101\u0101\u0501\u0501",
                                                         "kl m", "nop1", "\\qrs2", null,
                                                         "3tuv\'", "wx4yz", "\ud720\ud721");
         ColumnVector e_upper = ColumnVector.fromStrings("A", "B", "CD", "\u0480\u0480", "E\tF",
                                                         "G\nH", "IJ\"\u0100\u0100\u0500\u0500",
                                                         "KL M", "NOP1", "\\QRS2", null,
                                                         "3TUV\'", "WX4YZ", "\ud720\ud721");
         ColumnVector lower = v.lower();
         ColumnVector upper = v.upper();) {
      assertColumnsAreEqual(lower, e_lower);
      assertColumnsAreEqual(upper, e_upper);
    }
  }

  @Test
  void testStringConcat() {
    try (ColumnVector v = ColumnVector.fromStrings("a", "B", "cd", "\u0480\u0481", "E\tf",
                                                   "g\nH", "IJ\"\u0100\u0101\u0500\u0501",
                                                   "kl m", "Nop1", "\\qRs2", "3tuV\'",
                                                   "wX4Yz", "\ud720\ud721");
         ColumnVector e_concat = ColumnVector.fromStrings("aa", "BB", "cdcd",
                                                   "\u0480\u0481\u0480\u0481", "E\tfE\tf", "g\nHg\nH",
                                                   "IJ\"\u0100\u0101\u0500\u0501IJ\"\u0100\u0101\u0500\u0501",
                                                   "kl mkl m", "Nop1Nop1", "\\qRs2\\qRs2", "3tuV\'3tuV\'",
                                                   "wX4YzwX4Yz", "\ud720\ud721\ud720\ud721");
         Scalar emptyString = Scalar.fromString("");
         ColumnVector concat = ColumnVector.stringConcatenate(emptyString, emptyString,
                                                              v, v)) {
      assertColumnsAreEqual(concat, e_concat);
    }
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("B", "cd", "\u0480\u0481", "E\tf");
           ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4);
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, emptyString,
                                                                sv, cv)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv1 = ColumnVector.fromStrings("a", "B", "cd");
           ColumnVector sv2 = ColumnVector.fromStrings("a", "B");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, emptyString,
                                                                sv1, sv2)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, emptyString,
                                                                sv)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           Scalar nullString = Scalar.fromString(null);
           ColumnVector concat = ColumnVector.stringConcatenate(nullString, emptyString,
                                                                sv, sv)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(null, emptyString,
                                                                sv, sv)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, null,
                                                                sv, sv)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, emptyString,
                                                                sv, null)) {}
    });
  }

  @Test
  void testStringConcatWithNulls() {
    try (ColumnVector v = ColumnVector.fromStrings("a", "B", "cd", "\u0480\u0481", "E\tf",
                                                   "g\nH", "IJ\"\u0100\u0101\u0500\u0501",
                                                   "kl m", "Nop1", "\\qRs2", null,
                                                   "3tuV\'", "wX4Yz", "\ud720\ud721");
         ColumnVector e_concat = ColumnVector.fromStrings("aa", "BB", "cdcd",
                                                   "\u0480\u0481\u0480\u0481", "E\tfE\tf", "g\nHg\nH",
                                                   "IJ\"\u0100\u0101\u0500\u0501IJ\"\u0100\u0101\u0500\u0501",
                                                   "kl mkl m", "Nop1Nop1", "\\qRs2\\qRs2", "NULLNULL",
                                                   "3tuV\'3tuV\'", "wX4YzwX4Yz", "\ud720\ud721\ud720\ud721");
          Scalar emptyString = Scalar.fromString("");
          Scalar nullSubstitute = Scalar.fromString("NULL");
         ColumnVector concat = ColumnVector.stringConcatenate(emptyString, nullSubstitute, v, v)) {
      assertColumnsAreEqual(concat, e_concat);
    }
  }

  @Test
  void testStringConcatSeparators() {
    try (ColumnVector sv1 = ColumnVector.fromStrings("a", "B", "cd", "\u0480\u0481", "E\tf", null, null, "\\G\u0100");
         ColumnVector sv2 = ColumnVector.fromStrings("b", "C", "\u0500\u0501", "x\nYz", null, null, "", null);
         ColumnVector e_concat = ColumnVector.fromStrings("aA1\t\ud721b", "BA1\t\ud721C", "cdA1\t\ud721\u0500\u0501",
                                                          "\u0480\u0481A1\t\ud721x\nYz", null, null, null, null);
         Scalar separatorString = Scalar.fromString("A1\t\ud721");
         Scalar nullString = Scalar.fromString(null);
         ColumnVector concat = ColumnVector.stringConcatenate(separatorString, nullString,
                                                              sv1, sv2)) {
      assertColumnsAreEqual(concat, e_concat);
    }
  }

  @Test
  void testWindowStatic() {
    WindowOptions options = WindowOptions.builder().window(1, 1)
        .minPeriods(2).build();
    try (ColumnVector v1 = ColumnVector.fromInts(5, 4, 7, 6, 8)) {
      try (ColumnVector expected = ColumnVector.fromInts(9, 16, 17, 21, 14);
           ColumnVector result = v1.rollingWindow(AggregateOp.SUM, options)) {
        assertColumnsAreEqual(expected, result);
      }

      try (ColumnVector expected = ColumnVector.fromInts(4, 4, 4, 6, 6);
           ColumnVector result = v1.rollingWindow(AggregateOp.MIN, options)) {
        assertColumnsAreEqual(expected, result);
      }

      try (ColumnVector expected = ColumnVector.fromInts(5, 7, 7, 8, 8);
           ColumnVector result = v1.rollingWindow(AggregateOp.MAX, options)) {
        assertColumnsAreEqual(expected, result);
      }

      // The rolling window produces the same result type as the input
      try (ColumnVector expected = ColumnVector.fromInts(4, 5, 5, 7, 7);
           ColumnVector result = v1.rollingWindow(AggregateOp.MEAN, options)) {
        assertColumnsAreEqual(expected, result);
      }

      try (ColumnVector expected = ColumnVector.fromInts(2, 3, 3, 3, 2);
           ColumnVector result = v1.rollingWindow(AggregateOp.COUNT, options)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testWindowDynamicNegative() {
    try (ColumnVector precedingCol = ColumnVector.fromInts(2, 2, 2, 3, 3);
         ColumnVector followingCol = ColumnVector.fromInts(-1, -1, -1, -1, 0)) {
      WindowOptions window = WindowOptions.builder()
          .minPeriods(2).window(precedingCol, followingCol).build();
      try (ColumnVector v1 = ColumnVector.fromInts(5, 4, 7, 6, 8);
           ColumnVector expected = ColumnVector.fromBoxedInts(null, null, 9, 16, 25);
           ColumnVector result = v1.rollingWindow(AggregateOp.SUM, window)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testWindowLag() {
    //TODO negative only works for ColumnVectors.  We need to file something to make it work for
    // static too
    try (ColumnVector precedingCol = ColumnVector.fromInts(1, 1, 1, 1, 1);
         ColumnVector followingCol = ColumnVector.fromInts(-1, -1, -1, -1, -1)) {
      WindowOptions window = WindowOptions.builder().minPeriods(1)
          .window(precedingCol, followingCol).build();
      try (ColumnVector v1 = ColumnVector.fromInts(5, 4, 7, 6, 8);
           ColumnVector expected = ColumnVector.fromBoxedInts(null, 5, 4, 7, 6);
           ColumnVector result = v1.rollingWindow(AggregateOp.MAX, window)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testWindowDynamic() {
    try (ColumnVector precedingCol = ColumnVector.fromInts(0, 1, 2, 0, 1);
         ColumnVector followingCol = ColumnVector.fromInts(2, 2, 2, 2, 2)) {
      WindowOptions window = WindowOptions.builder().minPeriods(2)
          .window(precedingCol, followingCol).build();
      try (ColumnVector v1 = ColumnVector.fromInts(5, 4, 7, 6, 8);
           ColumnVector expected = ColumnVector.fromInts(16, 22, 30, 14, 14);
           ColumnVector result = v1.rollingWindow(AggregateOp.SUM, window)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testWindowThrowsException() {
    try (ColumnVector arraywindowCol = ColumnVector.fromBoxedInts(1, 2, 3 ,1, 1)) {
      assertThrows(IllegalArgumentException.class, () -> WindowOptions.builder()
          .window(3, 2).minPeriods(3)
          .window(arraywindowCol, arraywindowCol).build());
    }
  }

  @Test
  void testFindAndReplaceAll() {
    try(ColumnVector vector = ColumnVector.fromInts(1, 4, 1, 5, 3, 3, 1, 2, 9, 8);
        ColumnVector oldValues = ColumnVector.fromInts(1, 4, 7); // 7 doesn't exist, nothing to replace
        ColumnVector replacedValues = ColumnVector.fromInts(7, 6, 1);
        ColumnVector expectedVector = ColumnVector.fromInts(7, 6, 7, 5, 3, 3, 7, 2, 9, 8);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
        assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllFloat() {
    try(ColumnVector vector = ColumnVector.fromFloats(1.0f, 4.2f, 1.3f, 5.7f, 3f, 3f, 1.0f, 2.6f, 0.9f, 8.3f);
        ColumnVector oldValues = ColumnVector.fromFloats(1.0f, 4.2f, 7); // 7 doesn't exist, nothing to replace
        ColumnVector replacedValues = ColumnVector.fromFloats(7.3f, 6.7f, 1.0f);
        ColumnVector expectedVector = ColumnVector.fromFloats(7.3f, 6.7f, 1.3f, 5.7f, 3f, 3f, 7.3f, 2.6f, 0.9f, 8.3f);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllTimeUnits() {
    try(ColumnVector vector = ColumnVector.timestampMicroSecondsFromLongs(1l, 1l, 2l, 8l);
        ColumnVector oldValues = ColumnVector.timestampMicroSecondsFromLongs(1l, 2l, 7l); // 7 dosn't exist, nothing to replace
        ColumnVector replacedValues = ColumnVector.timestampMicroSecondsFromLongs(9l, 4l, 0l);
        ColumnVector expectedVector = ColumnVector.timestampMicroSecondsFromLongs(9l, 9l, 4l, 8l);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllMixingTypes() {
    try(ColumnVector vector = ColumnVector.fromInts(1, 4, 1, 5, 3, 3, 1, 2, 9, 8);
        ColumnVector oldValues = ColumnVector.fromInts(1, 4, 7); // 7 doesn't exist, nothing to replace
        ColumnVector replacedValues = ColumnVector.fromFloats(7.0f, 6, 1)) {
      assertThrows(CudfException.class, () -> vector.findAndReplaceAll(oldValues, replacedValues));
    }
  }

  @Test
  void testFindAndReplaceAllStrings() {
    try(ColumnVector vector = ColumnVector.fromStrings("spark", "scala", "spark", "hello", "code");
        ColumnVector oldValues = ColumnVector.fromStrings("spark","code","hello");
        ColumnVector replacedValues = ColumnVector.fromStrings("sparked", "codec", "hi");
        ColumnVector expectedValues = ColumnVector.fromStrings("sparked", "scala", "sparked", "hi", "codec");
        ColumnVector cv = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedValues, cv);
    }
  }

  @Test
  void testFindAndReplaceAllWithNull() {
    try(ColumnVector vector = ColumnVector.fromBoxedInts(1, 4, 1, 5, 3, 3, 1, null, 9, 8);
        ColumnVector oldValues = ColumnVector.fromBoxedInts(1, 4, 8);
        ColumnVector replacedValues = ColumnVector.fromBoxedInts(7, 6, null);
        ColumnVector expectedVector = ColumnVector.fromBoxedInts(7, 6, 7, 5, 3, 3, 7, null, 9, null);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllNulllWithValue() {
    // null values cannot be replaced using findAndReplaceAll();
    try(ColumnVector vector = ColumnVector.fromBoxedInts(1, 4, 1, 5, 3, 3, 1, null, 9, 8);
        ColumnVector oldValues = ColumnVector.fromBoxedInts(1, 4, null);
        ColumnVector replacedValues = ColumnVector.fromBoxedInts(7, 6, 8)) {
      assertThrows(CudfException.class, () -> vector.findAndReplaceAll(oldValues, replacedValues));
    }
  }

  @Test
  void testFindAndReplaceAllFloatNan() {
    // Float.NaN != Float.NaN therefore it cannot be replaced
    try(ColumnVector vector = ColumnVector.fromFloats(1.0f, 4.2f, 1.3f, 5.7f, 3f, 3f, 1.0f, 2.6f, Float.NaN, 8.3f);
        ColumnVector oldValues = ColumnVector.fromFloats(1.0f, 4.2f, Float.NaN);
        ColumnVector replacedValues = ColumnVector.fromFloats(7.3f, 6.7f, 0);
        ColumnVector expectedVector = ColumnVector.fromFloats(7.3f, 6.7f, 1.3f, 5.7f, 3f, 3f, 7.3f, 2.6f, Float.NaN, 8.3f);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testFindAndReplaceAllWithFloatNan() {
    try(ColumnVector vector = ColumnVector.fromFloats(1.0f, 4.2f, 1.3f, 5.7f, 3f, 3f, 1.0f, 2.6f, Float.NaN, 8.3f);
        ColumnVector oldValues = ColumnVector.fromFloats(1.0f, 4.2f, 8.3f);
        ColumnVector replacedValues = ColumnVector.fromFloats(7.3f, Float.NaN, 0);
        ColumnVector expectedVector = ColumnVector.fromFloats(7.3f, Float.NaN, 1.3f, 5.7f, 3f, 3f, 7.3f, 2.6f, Float.NaN, 0);
        ColumnVector newVector = vector.findAndReplaceAll(oldValues, replacedValues)) {
      assertColumnsAreEqual(expectedVector, newVector);
    }
  }

  @Test
  void testCast() {
    int[] values = new int[]{1,3,4,5,2};
    long[] longValues = Arrays.stream(values).asLongStream().toArray();
    double[] doubleValues = Arrays.stream(values).asDoubleStream().toArray();
    byte[] byteValues = new byte[values.length];
    float[] floatValues = new float[values.length];
    short[] shortValues = new short[values.length];
    IntStream.range(0, values.length).forEach(i -> {
      byteValues[i] = (byte)values[i];
      floatValues[i] = (float)values[i];
      shortValues[i] = (short)values[i];
    });

    try (ColumnVector cv = ColumnVector.fromInts(values);
         ColumnVector expectedBytes = ColumnVector.fromBytes(byteValues);
         ColumnVector bytes = cv.asBytes();
         ColumnVector expectedFloats = ColumnVector.fromFloats(floatValues);
         ColumnVector floats = cv.asFloats();
         ColumnVector expectedDoubles = ColumnVector.fromDoubles(doubleValues);
         ColumnVector doubles = cv.asDoubles();
         ColumnVector expectedLongs = ColumnVector.fromLongs(longValues);
         ColumnVector longs = cv.asLongs();
         ColumnVector expectedShorts = ColumnVector.fromShorts(shortValues);
         ColumnVector shorts = cv.asShorts();
         ColumnVector expectedDays = ColumnVector.daysFromInts(values);
         ColumnVector days = cv.asTimestampDays();
         ColumnVector expectedUs = ColumnVector.timestampMicroSecondsFromLongs(longValues);
         ColumnVector us = cv.asTimestampMicroseconds();
         ColumnVector expectedNs = ColumnVector.timestampNanoSecondsFromLongs(longValues);
         ColumnVector ns = cv.asTimestampNanoseconds();
         ColumnVector expectedMs = ColumnVector.timestampMilliSecondsFromLongs(longValues);
         ColumnVector ms = cv.asTimestampMilliseconds();
         ColumnVector expectedS = ColumnVector.timestampSecondsFromLongs(longValues);
         ColumnVector s = cv.asTimestampSeconds();) {
      assertColumnsAreEqual(expectedBytes, bytes);
      assertColumnsAreEqual(expectedShorts, shorts);
      assertColumnsAreEqual(expectedLongs, longs);
      assertColumnsAreEqual(expectedDoubles, doubles);
      assertColumnsAreEqual(expectedFloats, floats);
      assertColumnsAreEqual(expectedDays, days);
      assertColumnsAreEqual(expectedUs, us);
      assertColumnsAreEqual(expectedMs, ms);
      assertColumnsAreEqual(expectedNs, ns);
      assertColumnsAreEqual(expectedS, s);
    }
  }

  @Test
  void testCastTimestampAsString() {
    final String[] TIMES_S_STRING = {
        "2018-07-04 12:00:00",
        "2023-01-25 07:32:12",
        "2018-07-04 12:00:00"};

    final long[] TIMES_S = {
        1530705600L,   //'2018-07-04 12:00:00'
        1674631932L,   //'2023-01-25 07:32:12'
        1530705600L};  //'2018-07-04 12:00:00'

    final String[] UNSUPPORTED_TIME_S_STRING = {"1965-10-26 14:01:12",
        "1960-02-06 19:22:11"};

    final long[] UNSUPPORTED_TIME_S = {-131968728L,   //'1965-10-26 14:01:12'
        -312439069L};   //'1960-02-06 19:22:11'

    final long[] TIMES_NS = {
        1530705600115254330L,   //'2018-07-04 12:00:00.115254330'
        1674631932929861604L,   //'2023-01-25 07:32:12.929861604'
        1530705600115254330L};  //'2018-07-04 12:00:00.115254330'

    final long[] UNSUPPORTED_TIME_NS = {-131968727761702469L};   //'1965-10-26 14:01:12.238297531'

    final String[] TIMES_NS_STRING = {
        "2018-07-04 12:00:00.115254330",
        "2023-01-25 07:32:12.929861604",
        "2018-07-04 12:00:00.115254330"};

    final String[] UNSUPPORTED_TIME_NS_STRING = {"1965-10-26 14:01:12.238297531"};

    // Seconds
    try (ColumnVector s_string_times = ColumnVector.fromStrings(TIMES_S_STRING);
         ColumnVector s_timestamps = ColumnVector.timestampSecondsFromLongs(TIMES_S);
         ColumnVector unsupported_s_string_times = ColumnVector.fromStrings(UNSUPPORTED_TIME_S_STRING);
         ColumnVector unsupported_s_timestamps = ColumnVector.timestampSecondsFromLongs(UNSUPPORTED_TIME_S);
         ColumnVector timestampsAsStrings = s_timestamps.asStrings("%Y-%m-%d %H:%M:%S")) {
      assertColumnsAreEqual(s_string_times, timestampsAsStrings);
      assertThrows(AssertionFailedError.class, () -> assertColumnsAreEqual(unsupported_s_string_times, unsupported_s_timestamps));
    }

    // Nanoseconds
    try (ColumnVector ns_string_times = ColumnVector.fromStrings(TIMES_NS_STRING);
         ColumnVector ns_timestamps = ColumnVector.timestampNanoSecondsFromLongs(TIMES_NS);
         ColumnVector unsupported_ns_string_times = ColumnVector.fromStrings(UNSUPPORTED_TIME_NS_STRING);
         ColumnVector unsupported_ns_timestamps = ColumnVector.timestampSecondsFromLongs(UNSUPPORTED_TIME_NS);
         ColumnVector timestampsAsStrings = ns_timestamps.asStrings("%Y-%m-%d %H:%M:%S.%f")) {
      assertColumnsAreEqual(ns_string_times, timestampsAsStrings);
      assertThrows(AssertionFailedError.class, () -> assertColumnsAreEqual(unsupported_ns_string_times, unsupported_ns_timestamps));
    }

  }

  @Test
  void testContainsScalar() {
    try (ColumnVector columnVector = ColumnVector.fromInts(1, 43, 42, 11, 2);
    Scalar s0 = Scalar.fromInt(3);
    Scalar s1 = Scalar.fromInt(43)) {
      assertFalse(columnVector.contains(s0));
      assertTrue(columnVector.contains(s1));
    }
  }

  @Test
  void testContainsVector() {
    try (ColumnVector columnVector = ColumnVector.fromBoxedInts(1, null, 43, 42, 11, 2);
         ColumnVector cv0 = ColumnVector.fromBoxedInts(1, 3, null, 11);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, false, false, true, false);
         ColumnVector result = columnVector.contains(cv0)) {
      assertColumnsAreEqual(expected, result);
    }
    try (ColumnVector columnVector = ColumnVector.fromStrings("1", "43", "42", "11", "2");
         ColumnVector cv0 = ColumnVector.fromStrings("1", "3", "11");
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, true, false);
         ColumnVector result = columnVector.contains(cv0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testStartAndEndsWith() {
    try (ColumnVector testStrings = ColumnVector.fromStrings("", null, "abCD", "1a\"\u0100B1", "a\"\u0100B1", "1a\"\u0100B",
                                      "1a\"\u0100B1\n\t\'", "1a\"\u0100B1\u0453\u1322\u5112", "1a\"\u0100B1Fg26",
                                      "1a\"\u0100B1\\\"\r1a\"\u0100B1", "1a\"\u0100B1\u0498\u1321\u51091a\"\u0100B1",
                                      "1a\"\u0100B1H2O11a\"\u0100B1", "1a\"\u0100B1\\\"\r1a\"\u0100B1",
                                      "\n\t\'1a\"\u0100B1", "\u0453\u1322\u51121a\"\u0100B1", "Fg261a\"\u0100B1");
         ColumnVector emptyStrings = ColumnVector.fromStrings();
         Scalar patternString = Scalar.fromString("1a\"\u0100B1");
         ColumnVector startsResult = testStrings.startsWith(patternString);
         ColumnVector endsResult = testStrings.endsWith(patternString);
         ColumnVector expectedStarts = ColumnVector.fromBoxedBooleans(false, null, false, true, false,
                                                                      false, true, true, true, true, true,
                                                                      true, true, false, false, false);
         ColumnVector expectedEnds = ColumnVector.fromBoxedBooleans(false, null, false, true, false,
                                                                    false, false, false, false, true, true,
                                                                    true, true, true, true, true);
         ColumnVector startsEmpty = emptyStrings.startsWith(patternString);
         ColumnVector endsEmpty = emptyStrings.endsWith(patternString);
         ColumnVector expectedEmpty = ColumnVector.fromBoxedBooleans()) {
      assertColumnsAreEqual(startsResult, expectedStarts);
      assertColumnsAreEqual(endsResult, expectedEnds);
      assertColumnsAreEqual(startsEmpty, expectedEmpty);
      assertColumnsAreEqual(endsEmpty, expectedEmpty);
    }
  }

  @Test
  void testStartAndEndsWithThrowsException() {
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = sv.startsWith(emptyString)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = sv.endsWith(emptyString)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString(null);
           ColumnVector concat = sv.startsWith(emptyString)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString(null);
           ColumnVector concat = sv.endsWith(emptyString)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           ColumnVector concat = sv.startsWith(null)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           ColumnVector concat = sv.endsWith(null)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar intScalar = Scalar.fromInt(1);
           ColumnVector concat = sv.startsWith(intScalar)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar intScalar = Scalar.fromInt(1);
           ColumnVector concat = sv.endsWith(intScalar)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector v = ColumnVector.fromInts(1, 43, 42, 11, 2);
           Scalar patternString = Scalar.fromString("a");
           ColumnVector concat = v.startsWith(patternString)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector v = ColumnVector.fromInts(1, 43, 42, 11, 2);
           Scalar patternString = Scalar.fromString("a");
           ColumnVector concat = v.endsWith(patternString)) {}
    });
  }

  @Test
  void testStringLocate() {
    try(ColumnVector v = ColumnVector.fromStrings("Héllo", "thésé", null, "\r\ud720é\ud721", "ARé",
                                                  "\\THE\t8\ud720", "tést strings", "", "éé");
        ColumnVector e_locate1 = ColumnVector.fromBoxedInts(1, 2, null, 2, 2, -1, 1, -1, 0);
        ColumnVector e_locate2 = ColumnVector.fromBoxedInts(-1, 2, null, -1, -1, -1, 1, -1, -1);
        ColumnVector e_locate3 = ColumnVector.fromBoxedInts(-1, -1, null, 1, -1, 6, -1, -1, -1);
        Scalar pattern1 = Scalar.fromString("é");
        Scalar pattern2 = Scalar.fromString("és");
        Scalar pattern3 = Scalar.fromString("\ud720");
        ColumnVector locate1 = v.stringLocate(pattern1, 0, -1);
        ColumnVector locate2 = v.stringLocate(pattern2, 0, -1);
        ColumnVector locate3 = v.stringLocate(pattern3, 0, -1)) {
      assertColumnsAreEqual(locate1, e_locate1);
      assertColumnsAreEqual(locate2, e_locate2);
      assertColumnsAreEqual(locate3, e_locate3);
    }
  }

  @Test
  void testStringLocateOffsets() {
    try(ColumnVector v = ColumnVector.fromStrings("Héllo", "thésé", null, "\r\ud720é\ud721", "ARé",
                                                  "\\THE\t8\ud720", "tést strings", "", "éé");
        Scalar pattern = Scalar.fromString("é");
        ColumnVector e_empty = ColumnVector.fromBoxedInts(-1, -1, null, -1, -1, -1, -1, -1, -1);
        ColumnVector e_start = ColumnVector.fromBoxedInts(-1, 2, null, 2, 2, -1, -1, -1, -1);
        ColumnVector e_end = ColumnVector.fromBoxedInts(1, -1, null, -1, -1, -1, 1, -1, 0);
        ColumnVector locate_empty = v.stringLocate(pattern, 13, -1);
        ColumnVector locate_start = v.stringLocate(pattern, 2, -1);
        ColumnVector locate_end = v.stringLocate(pattern, 0, 2)) {
      assertColumnsAreEqual(locate_empty, e_empty);
      assertColumnsAreEqual(locate_start, e_start);
      assertColumnsAreEqual(locate_end, e_end);
    }
  }

  @Test
  void testStringLocateThrowsException() {
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("Héllo", "thésé", null, "ARé", "tést strings");
           ColumnVector locate = cv.stringLocate(null, 0, -1)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("Héllo", "thésé", null, "ARé", "tést strings");
           Scalar pattern = Scalar.fromString(null);
           ColumnVector locate = cv.stringLocate(pattern, 0, -1)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("Héllo", "thésé", null, "ARé", "tést strings");
           Scalar intScalar = Scalar.fromInt(1);
           ColumnVector locate = cv.stringLocate(intScalar, 0, -1)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("Héllo", "thésé", null, "ARé", "tést strings");
           Scalar pattern = Scalar.fromString("");
           ColumnVector locate = cv.stringLocate(pattern, 0, -1)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("Héllo", "thésé", null, "ARé", "tést strings");
           Scalar pattern = Scalar.fromString("é");
           ColumnVector locate = cv.stringLocate(pattern, -2, -1)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("Héllo", "thésé", null, "ARé", "tést strings");
           Scalar pattern = Scalar.fromString("é");
           ColumnVector locate = cv.stringLocate(pattern, 2, 1)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromInts(1, 43, 42, 11, 2);
           Scalar pattern = Scalar.fromString("é");
           ColumnVector concat = cv.stringLocate(pattern, 0, -1)) {}
    });
  }
}
