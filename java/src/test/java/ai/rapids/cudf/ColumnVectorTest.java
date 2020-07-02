/*
 *
 *  Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import static ai.rapids.cudf.QuantileMethod.HIGHER;
import static ai.rapids.cudf.QuantileMethod.LINEAR;
import static ai.rapids.cudf.QuantileMethod.LOWER;
import static ai.rapids.cudf.QuantileMethod.MIDPOINT;
import static ai.rapids.cudf.QuantileMethod.NEAREST;
import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;
import static ai.rapids.cudf.TableTest.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class ColumnVectorTest extends CudfTestBase {

  public static final double PERCENTAGE = 0.0001;

  // IEEE 754 NaN values
  static final float POSITIVE_FLOAT_NAN_LOWER_RANGE = Float.intBitsToFloat(0x7f800001);
  static final float POSITIVE_FLOAT_NAN_UPPER_RANGE = Float.intBitsToFloat(0x7fffffff);
  static final float NEGATIVE_FLOAT_NAN_LOWER_RANGE = Float.intBitsToFloat(0xff800001);
  static final float NEGATIVE_FLOAT_NAN_UPPER_RANGE = Float.intBitsToFloat(0xffffffff);

  static final double POSITIVE_DOUBLE_NAN_LOWER_RANGE = Double.longBitsToDouble(0x7ff0000000000001L);
  static final double POSITIVE_DOUBLE_NAN_UPPER_RANGE = Double.longBitsToDouble(0x7fffffffffffffffL);
  static final double NEGATIVE_DOUBLE_NAN_LOWER_RANGE = Double.longBitsToDouble(0xfff0000000000001L);
  static final double NEGATIVE_DOUBLE_NAN_UPPER_RANGE = Double.longBitsToDouble(0xffffffffffffffffL);

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

  static String cuda = "__device__ inline void f(" +
      "int* output," +
      "int input" +
      "){" +
      "*output = input*input - input;" +
      "}";

  @Test
  void testTransformVector() {
    try (ColumnVector cv = ColumnVector.fromBoxedInts(2,3,null,4);
         ColumnVector cv1 = cv.transform(ptx, true);
         ColumnVector cv2 = cv.transform(cuda, false);
         ColumnVector expected = ColumnVector.fromBoxedInts(2*2-2, 3*3-3, null, 4*4-4)) {
      TableTest.assertColumnsAreEqual(expected, cv1);
      TableTest.assertColumnsAreEqual(expected, cv2);
    }
  }

  @Test
  void testClampDouble() {
    try (ColumnVector cv = ColumnVector.fromDoubles(2.33d, 32.12d, -121.32d, 0.0d, 0.00001d,
        Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NaN);
         Scalar num = Scalar.fromDouble(0);
         Scalar loReplace = Scalar.fromDouble(-1);
         Scalar hiReplace = Scalar.fromDouble(1);
         ColumnVector result = cv.clamp(num, loReplace, num, hiReplace);
         ColumnVector expected = ColumnVector.fromDoubles(1.0d, 1.0d, -1.0d, 0.0d, 1.0d, -1.0d,
             1.0d, Double.NaN)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testClampFloat() {
    try (ColumnVector cv = ColumnVector.fromBoxedFloats(2.33f, 32.12f, null, -121.32f, 0.0f, 0.00001f, Float.NEGATIVE_INFINITY,
        Float.POSITIVE_INFINITY, Float.NaN);
         Scalar num = Scalar.fromFloat(0);
         Scalar loReplace = Scalar.fromFloat(-1);
         Scalar hiReplace = Scalar.fromFloat(1);
         ColumnVector result = cv.clamp(num, loReplace, num, hiReplace);
         ColumnVector expected = ColumnVector.fromBoxedFloats(1.0f, 1.0f, null, -1.0f, 0.0f, 1.0f, -1.0f, 1.0f, Float.NaN)) {
     assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testClampLong() {
    try (ColumnVector cv = ColumnVector.fromBoxedLongs(1l, 3l, 6l, -2l, 23l, -0l, -90l, null);
         Scalar num = Scalar.fromLong(0);
         Scalar loReplace = Scalar.fromLong(-1);
         Scalar hiReplace = Scalar.fromLong(1);
         ColumnVector result = cv.clamp(num, loReplace, num, hiReplace);
         ColumnVector expected = ColumnVector.fromBoxedLongs(1l, 1l, 1l, -1l, 1l, 0l, -1l, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testClampShort() {
    try (ColumnVector cv = ColumnVector.fromShorts(new short[]{1, 3, 6, -2, 23, -0, -90});
         Scalar lo = Scalar.fromShort((short)1);
         Scalar hi = Scalar.fromShort((short)2);
         ColumnVector result = cv.clamp(lo, hi);
         ColumnVector expected = ColumnVector.fromShorts(new short[]{1, 2, 2, 1, 2, 1, 1})) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testClampInt() {
      try (ColumnVector cv = ColumnVector.fromInts(1, 3, 6, -2, 23, -0, -90);
           Scalar num = Scalar.fromInt(0);
           Scalar hiReplace = Scalar.fromInt(1);
           Scalar loReplace = Scalar.fromInt(-1);
           ColumnVector result = cv.clamp(num, loReplace, num, hiReplace);
           ColumnVector expected = ColumnVector.fromInts(1, 1, 1, -1, 1, 0, -1)) {
          assertColumnsAreEqual(expected, result);
      }
  }

 @Test
  void testStringCreation() {
    try (ColumnVector cv = ColumnVector.fromStrings("d", "sd", "sde", null, "END");
         HostColumnVector host = cv.copyToHost();
         ColumnVector backAgain = host.copyToDevice()) {
      TableTest.assertColumnsAreEqual(cv, backAgain);
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
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2);
         ColumnVector expected = ColumnVector.fromInts(1, 2, 3, 4, 5, 6, 7, 8, 9)) {
      TableTest.assertColumnsAreEqual(expected, v);
    }
  }

  @Test
  void testConcatWithNulls() {
    try (ColumnVector v0 = ColumnVector.fromDoubles(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromDoubles(5, 6, 7);
         ColumnVector v2 = ColumnVector.fromBoxedDoubles(null, 9.0);
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(1., 2., 3., 4., 5., 6., 7., null, 9.)) {
      TableTest.assertColumnsAreEqual(expected, v);
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
  void testNormalizeNANsAndZeros() {
    // Must check boundaries of NaN representation, as described in javadoc for Double#longBitsToDouble.
    // @see java.lang.Double#longBitsToDouble
    // <quote>
    //      If the argument is any value in the range 0x7ff0000000000001L through 0x7fffffffffffffffL,
    //      or in the range 0xfff0000000000001L through 0xffffffffffffffffL, the result is a NaN.
    // </quote>
    final double MIN_PLUS_NaN  = Double.longBitsToDouble(0x7ff0000000000001L);
    final double MAX_PLUS_NaN  = Double.longBitsToDouble(0x7fffffffffffffffL);
    final double MAX_MINUS_NaN = Double.longBitsToDouble(0xfff0000000000001L);
    final double MIN_MINUS_NaN = Double.longBitsToDouble(0xffffffffffffffffL);

    Double[]  ins = new Double[] {0.0, -0.0, Double.NaN, MIN_PLUS_NaN, MAX_PLUS_NaN, MIN_MINUS_NaN, MAX_MINUS_NaN, null};
    Double[] outs = new Double[] {0.0,  0.0, Double.NaN,   Double.NaN,   Double.NaN,    Double.NaN,    Double.NaN, null};

    try (ColumnVector input = ColumnVector.fromBoxedDoubles(ins);
         ColumnVector expectedColumn = ColumnVector.fromBoxedDoubles(outs);
         ColumnVector normalizedColumn = input.normalizeNANsAndZeros()) {
      try (HostColumnVector expected = expectedColumn.copyToHost();
           HostColumnVector normalized = normalizedColumn.copyToHost()) {
        for (int i = 0; i<input.getRowCount(); ++i) {
          if (expected.isNull(i)) {
            assertTrue(normalized.isNull(i));
          }
          else {
            assertEquals(
                    Double.doubleToRawLongBits(expected.getDouble(i)),
                    Double.doubleToRawLongBits(normalized.getDouble(i))
            );
          }
        }
      }
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
  void testSequenceInt() {
    try (Scalar zero = Scalar.fromInt(0);
         Scalar one = Scalar.fromInt(1);
         Scalar negOne = Scalar.fromInt(-1);
         Scalar nulls = Scalar.fromNull(DType.INT32)) {

      try (
          ColumnVector cv = ColumnVector.sequence(zero, 5);
          ColumnVector expected = ColumnVector.fromInts(0, 1, 2, 3, 4)) {
        assertColumnsAreEqual(expected, cv);
      }


      try (ColumnVector cv = ColumnVector.sequence(one, negOne,6);
           ColumnVector expected = ColumnVector.fromInts(1, 0, -1, -2, -3, -4)) {
        assertColumnsAreEqual(expected, cv);
      }

      try (ColumnVector cv = ColumnVector.sequence(zero, 0);
           ColumnVector expected = ColumnVector.fromInts()) {
        assertColumnsAreEqual(expected, cv);
      }

      assertThrows(IllegalArgumentException.class, () -> {
        try (ColumnVector cv = ColumnVector.sequence(nulls, 5)) { }
      });

      assertThrows(CudfException.class, () -> {
        try (ColumnVector cv = ColumnVector.sequence(zero, -3)) { }
      });
    }
  }

  @Test
  void testSequenceDouble() {
    try (Scalar zero = Scalar.fromDouble(0.0);
         Scalar one = Scalar.fromDouble(1.0);
         Scalar negOneDotOne = Scalar.fromDouble(-1.1)) {

      try (
          ColumnVector cv = ColumnVector.sequence(zero, 5);
          ColumnVector expected = ColumnVector.fromDoubles(0, 1, 2, 3, 4)) {
        assertColumnsAreEqual(expected, cv);
      }


      try (ColumnVector cv = ColumnVector.sequence(one, negOneDotOne,6);
           ColumnVector expected = ColumnVector.fromDoubles(1, -0.1, -1.2, -2.3, -3.4, -4.5)) {
        assertColumnsAreEqual(expected, cv);
      }

      try (ColumnVector cv = ColumnVector.sequence(zero, 0);
           ColumnVector expected = ColumnVector.fromDoubles()) {
        assertColumnsAreEqual(expected, cv);
      }
    }
  }

  @Test
  void testSequenceOtherTypes() {
    assertThrows(CudfException.class, () -> {
      try (Scalar s = Scalar.fromString("0");
      ColumnVector cv = ColumnVector.sequence(s, s, 5)) {}
    });

    assertThrows(CudfException.class, () -> {
      try (Scalar s = Scalar.fromBool(false);
           ColumnVector cv = ColumnVector.sequence(s, s, 5)) {}
    });

    assertThrows(CudfException.class, () -> {
      try (Scalar s = Scalar.timestampDaysFromInt(100);
           ColumnVector cv = ColumnVector.sequence(s, s, 5)) {}
    });
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
        case UINT8:
          s = Scalar.fromUnsignedByte((byte) 254);
          break;
        case INT16:
          s = Scalar.fromShort((short) 12345);
          break;
        case UINT16:
          s = Scalar.fromUnsignedShort((short) 65432);
          break;
        case INT32:
          s = Scalar.fromInt(123456789);
          break;
        case UINT32:
          s = Scalar.fromUnsignedInt(0xfedcba98);
          break;
        case INT64:
          s = Scalar.fromLong(1234567890123456789L);
          break;
        case UINT64:
          s = Scalar.fromUnsignedLong(0xfedcba9876543210L);
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
  void testGetNativeView() {
    try (ColumnVector cv = ColumnVector.fromInts(1, 3, 4, 5)) {
      //not a real test whats being returned is a view but this is the best we can do
      assertNotEquals(0L, cv.getNativeView());
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
        case UINT8: {
          byte v = (byte) 254;
          s = Scalar.fromUnsignedByte(v);
          expected = ColumnVector.fromBoxedUnsignedBytes(v, v, v, v);
          break;
        }
        case INT16: {
          short v = (short) 12345;
          s = Scalar.fromShort(v);
          expected = ColumnVector.fromBoxedShorts(v, v, v, v);
          break;
        }
        case UINT16: {
          short v = (short) 0x89ab;
          s = Scalar.fromUnsignedShort(v);
          expected = ColumnVector.fromBoxedUnsignedShorts(v, v, v, v);
          break;
        }
        case INT32: {
          int v = 123456789;
          s = Scalar.fromInt(v);
          expected = ColumnVector.fromBoxedInts(v, v, v, v);
          break;
        }
        case UINT32: {
          int v = 0x89abcdef;
          s = Scalar.fromUnsignedInt(v);
          expected = ColumnVector.fromBoxedUnsignedInts(v, v, v, v);
          break;
        }
        case INT64: {
          long v = 1234567890123456789L;
          s = Scalar.fromLong(v);
          expected = ColumnVector.fromBoxedLongs(v, v, v, v);
          break;
        }
        case UINT64: {
          long v = 0xfedcba9876543210L;
          s = Scalar.fromUnsignedLong(v);
          expected = ColumnVector.fromBoxedUnsignedLongs(v, v, v, v);
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
           ColumnVector c = ColumnVector.fromScalar(s, rowCount);
           HostColumnVector hc = c.copyToHost()) {
        assertEquals(type, c.getType());
        assertEquals(rowCount, c.getRowCount());
        assertEquals(rowCount, c.getNullCount());
        for (int i = 0; i < rowCount; ++i) {
          assertTrue(hc.isNull(i));
        }
      }
    }
  }

  @Test
  void testFromScalarNullByte() {
    int numNulls = 3000;
    try (Scalar s = Scalar.fromNull(DType.INT8);
         ColumnVector tmp = ColumnVector.fromScalar(s, numNulls);
         HostColumnVector input = tmp.copyToHost()) {
      assertEquals(numNulls, input.getRowCount());
      assertEquals(input.getNullCount(), numNulls);
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

    try (ColumnVector cv = ColumnVector.fromBoxedInts(-1, 0, 1, 1, 2, 3, 4, 6, 7, 9)) {
      for (int i = 0 ; i < methods.length ; i++) {
        try (ColumnVector result = cv.quantile(methods[i], quantiles);
             HostColumnVector hostResult = result.copyToHost()) {
          double[] expected = exactExpected[i];
          assertEquals(expected.length, hostResult.getRowCount());
          for (int j = 0; j < expected.length; j++) {
            assertEqualsWithinPercentage(expected[j], hostResult.getDouble(j), PERCENTAGE, methods[i] + " " + quantiles[j]);
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

    try (ColumnVector cv = ColumnVector.fromBoxedDoubles(-1.01, 0.15, 0.8, 1.11, 2.13, 3.4, 4.17, 5.7, 6.8)) {
      for (int i = 0 ; i < methods.length ; i++) {
        try (ColumnVector result = cv.quantile(methods[i], quantiles);
             HostColumnVector hostResult = result.copyToHost()) {
          double[] expected = exactExpected[i];
          assertEquals(expected.length, hostResult.getRowCount());
          for (int j = 0; j < expected.length; j++) {
            assertEqualsWithinPercentage(expected[j], hostResult.getDouble(j), PERCENTAGE, methods[i] + " " + quantiles[j]);
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
          try (HostColumnVector slice = slices[sliceIndex].copyToHost()) {
            assertEquals(expectedSlice[sliceIndex].length, slices[sliceIndex].getRowCount());
            IntStream.range(0, expectedSlice[sliceIndex].length).forEach(rowCount -> {
              if (expectedSlice[sliceIndex][rowCount] == null) {
                assertTrue(slice.isNull(rowCount));
              } else {
                assertEquals(expectedSlice[sliceIndex][rowCount],
                    slice.getInt(rowCount));
              }
            });
          }
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
          try (HostColumnVector slice = slices[sliceIndex].copyToHost()) {
            assertEquals(expectedSlice[sliceIndex].length, slices[sliceIndex].getRowCount());
            IntStream.range(0, expectedSlice[sliceIndex].length).forEach(rowCount -> {
              if (expectedSlice[sliceIndex][rowCount] == null) {
                assertTrue(slice.isNull(rowCount));
              } else {
                assertEquals(expectedSlice[sliceIndex][rowCount],
                    slice.getJavaString(rowCount));
              }
            });
          }
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
          try (HostColumnVector subVec = splits[splitIndex].copyToHost()) {
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
  void testTrimStringsWhiteSpace() {
    try (ColumnVector cv = ColumnVector.fromStrings(" 123", "123 ", null, " 123 ", "\t\t123\n\n");
         ColumnVector trimmed = cv.strip();
         ColumnVector expected = ColumnVector.fromStrings("123", "123", null, "123", "123")) {
      TableTest.assertColumnsAreEqual(expected, trimmed);
    }
  }

  @Test
  void testTrimStrings() {
    try (ColumnVector cv = ColumnVector.fromStrings("123", "123 ", null, "1231", "\t\t123\n\n");
         Scalar one = Scalar.fromString(" 1");
         ColumnVector trimmed = cv.strip(one);
         ColumnVector expected = ColumnVector.fromStrings("23", "23", null, "23", "\t\t123\n\n")) {
      TableTest.assertColumnsAreEqual(expected, trimmed);
    }
  }

  @Test
  void testLeftTrimStringsWhiteSpace() {
    try (ColumnVector cv = ColumnVector.fromStrings(" 123", "123 ", null, " 123 ", "\t\t123\n\n");
         ColumnVector trimmed = cv.lstrip();
         ColumnVector expected = ColumnVector.fromStrings("123", "123 ", null, "123 ", "123\n\n")) {
      TableTest.assertColumnsAreEqual(expected, trimmed);
    }
  }

  @Test
  void testLeftTrimStrings() {
    try (ColumnVector cv = ColumnVector.fromStrings("123", " 123 ", null, "1231", "\t\t123\n\n");
         Scalar one = Scalar.fromString(" 1");
         ColumnVector trimmed = cv.lstrip(one);
         ColumnVector expected = ColumnVector.fromStrings("23", "23 ", null, "231", "\t\t123\n\n")) {
      TableTest.assertColumnsAreEqual(expected, trimmed);
    }
  }

  @Test
  void testRightTrimStringsWhiteSpace() {
    try (ColumnVector cv = ColumnVector.fromStrings(" 123", "123 ", null, " 123 ", "\t\t123\n\n");
         ColumnVector trimmed = cv.rstrip();
         ColumnVector expected = ColumnVector.fromStrings(" 123", "123", null, " 123", "\t\t123")) {
      TableTest.assertColumnsAreEqual(expected, trimmed);
    }
  }

  @Test
  void testRightTrimStrings() {
    try (ColumnVector cv = ColumnVector.fromStrings("123", "123 ", null, "1231 ", "\t\t123\n\n");
         Scalar one = Scalar.fromString(" 1");
         ColumnVector trimmed = cv.rstrip(one);
         ColumnVector expected = ColumnVector.fromStrings("123", "123", null, "123", "\t\t123\n\n")) {
      TableTest.assertColumnsAreEqual(expected, trimmed);
    }
  }

  @Test
  void testTrimStringsThrowsException() {
    assertThrows(CudfException.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("123", "123 ", null, "1231", "\t\t123\n\n");
           Scalar nullStr =  Scalar.fromString(null);
           ColumnVector trimmed = cv.strip(nullStr)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("123", "123 ", null, "1231", "\t\t123\n\n");
           Scalar one = Scalar.fromInt(1);
           ColumnVector trimmed = cv.strip(one)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("123", "123 ", null, "1231", "\t\t123\n\n");
           ColumnVector result = cv.strip(null)) {}
    });
  }

  @Test
  void testAppendStrings() {
    try (HostColumnVector cv = HostColumnVector.build(10, 0, (b) -> {
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
      ColumnVector lengths = cv.getCharLengths();
      ColumnVector expected = ColumnVector.fromBoxedInts(1, 2, null, 3, 4)) {
      TableTest.assertColumnsAreEqual(expected, lengths);
    }
  }

  @Test
  void testGetByteCount() {
    try (ColumnVector cv = ColumnVector.fromStrings("1", "12", "123", null, "1234");
         ColumnVector byteLengthVector = cv.getByteCount();
         ColumnVector expected = ColumnVector.fromBoxedInts(1, 2, 3, null, 4)) {
      TableTest.assertColumnsAreEqual(expected, byteLengthVector);
    }
  }

  @Test
  void testEmptyStringColumnOpts() {
    try (ColumnVector cv = ColumnVector.fromStrings()) {
      try (ColumnVector len = cv.getCharLengths()) {
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
                                                              new ColumnVector[]{v, v})) {
      assertColumnsAreEqual(concat, e_concat);
    }
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("B", "cd", "\u0480\u0481", "E\tf");
           ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4);
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, emptyString,
                                                                new ColumnVector[]{sv, cv})) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv1 = ColumnVector.fromStrings("a", "B", "cd");
           ColumnVector sv2 = ColumnVector.fromStrings("a", "B");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, emptyString,
                                                                new ColumnVector[]{sv1, sv2})) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, emptyString,
                                                                new ColumnVector[]{sv})) {}
    });
    assertThrows(CudfException.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           Scalar nullString = Scalar.fromString(null);
           ColumnVector concat = ColumnVector.stringConcatenate(nullString, emptyString,
                                                                new ColumnVector[]{sv, sv})) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(null, emptyString,
                                                                new ColumnVector[]{sv, sv})) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, null,
                                                                new ColumnVector[]{sv, sv})) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector concat = ColumnVector.stringConcatenate(emptyString, emptyString,
                                                                new ColumnVector[]{sv, null})) {}
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
         ColumnVector concat = ColumnVector.stringConcatenate(emptyString, nullSubstitute,
                                                              new ColumnVector[]{v, v})) {
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
                                                              new ColumnVector[]{sv1, sv2})) {
      assertColumnsAreEqual(concat, e_concat);
    }
  }

  @Test
  void testWindowStatic() {
    WindowOptions options = WindowOptions.builder().window(2, 1)
        .minPeriods(2).build();
    try (ColumnVector v1 = ColumnVector.fromInts(5, 4, 7, 6, 8)) {
      try (ColumnVector expected = ColumnVector.fromLongs(9, 16, 17, 21, 14);
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
      try (ColumnVector expected = ColumnVector.fromDoubles(4.5, 16.0 / 3, 17.0 / 3, 7, 7);
           ColumnVector result = v1.rollingWindow(AggregateOp.MEAN, options)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testWindowStaticCounts() {
    WindowOptions options = WindowOptions.builder().window(2, 1)
            .minPeriods(2).build();
    try (ColumnVector v1 = ColumnVector.fromBoxedInts(5, 4, null, 6, 8)) {
      try (ColumnVector expected = ColumnVector.fromInts(2, 2, 2, 2, 2);
           ColumnVector result = v1.rollingWindow(AggregateOp.COUNT_VALID, options)) {
        assertColumnsAreEqual(expected, result);
      }
      try (ColumnVector expected = ColumnVector.fromInts(2, 3, 3, 3, 2);
           ColumnVector result = v1.rollingWindow(AggregateOp.COUNT_ALL, options)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testWindowDynamicNegative() {
    try (ColumnVector precedingCol = ColumnVector.fromInts(3, 3, 3, 4, 4);
         ColumnVector followingCol = ColumnVector.fromInts(-1, -1, -1, -1, 0)) {
      WindowOptions window = WindowOptions.builder()
          .minPeriods(2).window(precedingCol, followingCol).build();
      try (ColumnVector v1 = ColumnVector.fromInts(5, 4, 7, 6, 8);
           ColumnVector expected = ColumnVector.fromBoxedLongs(null, null, 9L, 16L, 25L);
           ColumnVector result = v1.rollingWindow(AggregateOp.SUM, window)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testWindowLag() {
    WindowOptions window = WindowOptions.builder().minPeriods(1)
        .window(2, -1).build();
    try (ColumnVector v1 = ColumnVector.fromInts(5, 4, 7, 6, 8);
         ColumnVector expected = ColumnVector.fromBoxedInts(null, 5, 4, 7, 6);
         ColumnVector result = v1.rollingWindow(AggregateOp.MAX, window)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testWindowDynamic() {
    try (ColumnVector precedingCol = ColumnVector.fromInts(1, 2, 3, 1, 2);
         ColumnVector followingCol = ColumnVector.fromInts(2, 2, 2, 2, 2)) {
      WindowOptions window = WindowOptions.builder().minPeriods(2)
          .window(precedingCol, followingCol).build();
      try (ColumnVector v1 = ColumnVector.fromInts(5, 4, 7, 6, 8);
           ColumnVector expected = ColumnVector.fromLongs(16, 22, 30, 14, 14);
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

      assertThrows(IllegalArgumentException.class, 
                   () -> arraywindowCol.rollingWindow(AggregateOp.SUM, 
                                                      WindowOptions.builder().window(2, 1).minPeriods(1).timestampColumnIndex(0).build()));
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
  void emptyStringColumnFindReplaceAll() {
    try (ColumnVector cv = ColumnVector.fromStrings(null, "A", "B", "C",   "");
         ColumnVector expected = ColumnVector.fromStrings(null, "A", "B", "C",   null);
         ColumnVector from = ColumnVector.fromStrings("");
         ColumnVector to = ColumnVector.fromStrings((String)null);
         ColumnVector replaced = cv.findAndReplaceAll(from, to)) {
      assertColumnsAreEqual(expected, replaced);
    }
  }

  @Test
  void testFixedWidthCast() {
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
         ColumnVector expectedUnsignedInts = ColumnVector.fromUnsignedInts(values);
         ColumnVector unsignedInts = cv.asUnsignedInts();
         ColumnVector expectedBytes = ColumnVector.fromBytes(byteValues);
         ColumnVector bytes = cv.asBytes();
         ColumnVector expectedUnsignedBytes = ColumnVector.fromUnsignedBytes(byteValues);
         ColumnVector unsignedBytes = cv.asUnsignedBytes();
         ColumnVector expectedFloats = ColumnVector.fromFloats(floatValues);
         ColumnVector floats = cv.asFloats();
         ColumnVector expectedDoubles = ColumnVector.fromDoubles(doubleValues);
         ColumnVector doubles = cv.asDoubles();
         ColumnVector expectedLongs = ColumnVector.fromLongs(longValues);
         ColumnVector longs = cv.asLongs();
         ColumnVector expectedUnsignedLongs = ColumnVector.fromUnsignedLongs(longValues);
         ColumnVector unsignedLongs = cv.asUnsignedLongs();
         ColumnVector expectedShorts = ColumnVector.fromShorts(shortValues);
         ColumnVector shorts = cv.asShorts();
         ColumnVector expectedUnsignedShorts = ColumnVector.fromUnsignedShorts(shortValues);
         ColumnVector unsignedShorts = cv.asUnsignedShorts();
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
      assertColumnsAreEqual(expectedUnsignedInts, unsignedInts);
      assertColumnsAreEqual(expectedBytes, bytes);
      assertColumnsAreEqual(expectedUnsignedBytes, unsignedBytes);
      assertColumnsAreEqual(expectedShorts, shorts);
      assertColumnsAreEqual(expectedUnsignedShorts, unsignedShorts);
      assertColumnsAreEqual(expectedLongs, longs);
      assertColumnsAreEqual(expectedUnsignedLongs, unsignedLongs);
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
  void testCastByteToString() {

    Byte[] byteValues = {1, 3, 45, -0, null, Byte.MIN_VALUE, Byte.MAX_VALUE};
    String[] stringByteValues = getStringArray(byteValues);

    testCastFixedWidthToStringsAndBack(DType.INT8, () -> ColumnVector.fromBoxedBytes(byteValues), () -> ColumnVector.fromStrings(stringByteValues));
  }

  @Test
  void testCastShortToString() {

    Short[] shortValues = {1, 3, 45, -0, null, Short.MIN_VALUE, Short.MAX_VALUE};
    String[] stringShortValues = getStringArray(shortValues);

    testCastFixedWidthToStringsAndBack(DType.INT16, () -> ColumnVector.fromBoxedShorts(shortValues), () -> ColumnVector.fromStrings(stringShortValues));
  }

  @Test
  void testCastIntToString() {
    Integer[] integerArray = {1, -2, 3, null, 8, Integer.MIN_VALUE, Integer.MAX_VALUE};
    String[] stringIntValues = getStringArray(integerArray);

    testCastFixedWidthToStringsAndBack(DType.INT32, () -> ColumnVector.fromBoxedInts(integerArray), () -> ColumnVector.fromStrings(stringIntValues));
  }

  @Test
  void testCastLongToString() {

    Long[] longValues = {null, 3l, 2l, -43l, null, Long.MIN_VALUE, Long.MAX_VALUE};
    String[] stringLongValues = getStringArray(longValues);

    testCastFixedWidthToStringsAndBack(DType.INT64, () -> ColumnVector.fromBoxedLongs(longValues), () -> ColumnVector.fromStrings(stringLongValues));
  }

  @Test
  void testCastFloatToString() {

    Float[] floatValues = {Float.NaN, null, 03f, -004f, 12f};
    String[] stringFloatValues = getStringArray(floatValues);

    testCastFixedWidthToStringsAndBack(DType.FLOAT32, () -> ColumnVector.fromBoxedFloats(floatValues), () -> ColumnVector.fromStrings(stringFloatValues));
  }

  @Test
  void testCastDoubleToString() {

    Double[] doubleValues = {Double.NaN, Double.NEGATIVE_INFINITY, 4d, 98d, null, Double.POSITIVE_INFINITY};
    //Creating the string array manually because of the way cudf converts POSITIVE_INFINITY to "Inf" instead of "INFINITY"
    String[] stringDoubleValues = {"NaN","-Inf", "4.0", "98.0", null, "Inf"};

    testCastFixedWidthToStringsAndBack(DType.FLOAT64, () -> ColumnVector.fromBoxedDoubles(doubleValues), () -> ColumnVector.fromStrings(stringDoubleValues));
  }

  @Test
  void testCastBoolToString() {

    Boolean[] booleans = {true, false, false};
    String[] stringBools = getStringArray(booleans);

    testCastFixedWidthToStringsAndBack(DType.BOOL8, () -> ColumnVector.fromBoxedBooleans(booleans), () -> ColumnVector.fromStrings(stringBools));
  }

  private static <T> String[] getStringArray(T[] input) {
    String[] result = new String[input.length];
    for (int i = 0 ; i < input.length ; i++) {
      if (input[i] == null) {
        result[i] = null;
      } else {
        result[i] = String.valueOf(input[i]);
      }
    }
    return result;
  }

  private static void testCastFixedWidthToStringsAndBack(DType type, Supplier<ColumnVector> fixedWidthSupplier,
                                                         Supplier<ColumnVector> stringColumnSupplier) {
    try (ColumnVector fixedWidthColumn = fixedWidthSupplier.get();
         ColumnVector stringColumn = stringColumnSupplier.get();
         ColumnVector fixedWidthCastedToString = fixedWidthColumn.castTo(DType.STRING);
         ColumnVector stringCastedToFixedWidth = stringColumn.castTo(type)) {
      assertColumnsAreEqual(stringColumn, fixedWidthCastedToString);
      assertColumnsAreEqual(fixedWidthColumn, stringCastedToFixedWidth);
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

    final long[] TIMES_NS = {
        1530705600115254330L,   //'2018-07-04 12:00:00.115254330'
        1674631932929861604L,   //'2023-01-25 07:32:12.929861604'
        1530705600115254330L};  //'2018-07-04 12:00:00.115254330'

    final String[] TIMES_NS_STRING = {
        "2018-07-04 12:00:00.115254330",
        "2023-01-25 07:32:12.929861604",
        "2018-07-04 12:00:00.115254330"};

    // all supported formats by cudf
    final String[] TIMES_NS_STRING_ALL = {
        "04::07::18::2018::12::00::00::115254330",
        "25::01::23::2023::07::32::12::929861604",
        "04::07::18::2018::12::00::00::115254330"};

    // Seconds
    try (ColumnVector s_string_times = ColumnVector.fromStrings(TIMES_S_STRING);
         ColumnVector s_timestamps = ColumnVector.timestampSecondsFromLongs(TIMES_S);
         ColumnVector timestampsAsStrings = s_timestamps.asStrings("%Y-%m-%d %H:%M:%S");
         ColumnVector timestampsAsStringsUsingDefaultFormat = s_timestamps.asStrings()) {
      assertColumnsAreEqual(s_string_times, timestampsAsStrings);
      assertColumnsAreEqual(timestampsAsStringsUsingDefaultFormat, timestampsAsStrings);
    }

    // Nanoseconds
    try (ColumnVector ns_string_times = ColumnVector.fromStrings(TIMES_NS_STRING);
         ColumnVector ns_timestamps = ColumnVector.timestampNanoSecondsFromLongs(TIMES_NS);
         ColumnVector ns_string_times_all = ColumnVector.fromStrings(TIMES_NS_STRING_ALL);
         ColumnVector allSupportedFormatsTimestampAsStrings = ns_timestamps.asStrings("%d::%m::%y::%Y::%H::%M::%S::%9f");
         ColumnVector timestampsAsStrings = ns_timestamps.asStrings("%Y-%m-%d %H:%M:%S.%9f")) {
      assertColumnsAreEqual(ns_string_times, timestampsAsStrings);
      assertColumnsAreEqual(allSupportedFormatsTimestampAsStrings, ns_string_times_all);
    }
  }

  @Test
  @Disabled("Negative timestamp values are not currently supported. " +
      "See github issue https://github.com/rapidsai/cudf/issues/3116 for details")
  void testCastNegativeTimestampAsString() {
    final String[] NEG_TIME_S_STRING = {"1965-10-26 14:01:12",
        "1960-02-06 19:22:11"};

    final long[] NEG_TIME_S = {-131968728L,   //'1965-10-26 14:01:12'
        -312439069L};   //'1960-02-06 19:22:11'

    final long[] NEG_TIME_NS = {-131968727761702469L};   //'1965-10-26 14:01:12.238297531'

    final String[] NEG_TIME_NS_STRING = {"1965-10-26 14:01:12.238297531"};

    // Seconds
    try (ColumnVector unsupported_s_string_times = ColumnVector.fromStrings(NEG_TIME_S_STRING);
         ColumnVector unsupported_s_timestamps = ColumnVector.timestampSecondsFromLongs(NEG_TIME_S)) {
      assertColumnsAreEqual(unsupported_s_string_times, unsupported_s_timestamps);
    }

    // Nanoseconds
    try (ColumnVector unsupported_ns_string_times = ColumnVector.fromStrings(NEG_TIME_NS_STRING);
         ColumnVector unsupported_ns_timestamps = ColumnVector.timestampSecondsFromLongs(NEG_TIME_NS)) {
      assertColumnsAreEqual(unsupported_ns_string_times, unsupported_ns_timestamps);
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
  void testStringOpsEmpty() {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd", null, "");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector found = sv.stringContains(emptyString);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, null, true)) {
          assertColumnsAreEqual(found, expected);
      }
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd", null, "");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector found = sv.startsWith(emptyString);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, null, true)) {
          assertColumnsAreEqual(found, expected);
      }
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd", null, "");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector found = sv.endsWith(emptyString);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, null, true)) {
          assertColumnsAreEqual(found, expected);
      }
      try (ColumnVector sv = ColumnVector.fromStrings("Héllo", "thésé", null, "ARé", "tést strings");
           Scalar emptyString = Scalar.fromString("");
           ColumnVector found = sv.stringLocate(emptyString, 0, -1);
           ColumnVector expected = ColumnVector.fromBoxedInts(0, 0, null, 0, 0)) {
          assertColumnsAreEqual(found, expected);
      }
  }

  @Test
  void testStringFindOperations() {
    try (ColumnVector testStrings = ColumnVector.fromStrings("", null, "abCD", "1a\"\u0100B1", "a\"\u0100B1", "1a\"\u0100B",
                                      "1a\"\u0100B1\n\t\'", "1a\"\u0100B1\u0453\u1322\u5112", "1a\"\u0100B1Fg26",
                                      "1a\"\u0100B1\\\"\r1a\"\u0100B1", "1a\"\u0100B1\u0498\u1321\u51091a\"\u0100B1",
                                      "1a\"\u0100B1H2O11a\"\u0100B1", "1a\"\u0100B1\\\"\r1a\"\u0100B1",
                                      "\n\t\'1a\"\u0100B1", "\u0453\u1322\u51121a\"\u0100B1", "Fg261a\"\u0100B1");
         ColumnVector emptyStrings = ColumnVector.fromStrings();
         Scalar patternString = Scalar.fromString("1a\"\u0100B1");
         ColumnVector startsResult = testStrings.startsWith(patternString);
         ColumnVector endsResult = testStrings.endsWith(patternString);
         ColumnVector containsResult = testStrings.stringContains(patternString);
         ColumnVector expectedStarts = ColumnVector.fromBoxedBooleans(false, null, false, true, false,
                                                                      false, true, true, true, true, true,
                                                                      true, true, false, false, false);
         ColumnVector expectedEnds = ColumnVector.fromBoxedBooleans(false, null, false, true, false,
                                                                    false, false, false, false, true, true,
                                                                    true, true, true, true, true);
         ColumnVector expectedContains = ColumnVector.fromBoxedBooleans(false, null, false, true, false, false,
                                                                        true, true, true, true, true,
                                                                        true, true, true, true, true);
         ColumnVector startsEmpty = emptyStrings.startsWith(patternString);
         ColumnVector endsEmpty = emptyStrings.endsWith(patternString);
         ColumnVector containsEmpty = emptyStrings.stringContains(patternString);
         ColumnVector expectedEmpty = ColumnVector.fromBoxedBooleans()) {
      assertColumnsAreEqual(startsResult, expectedStarts);
      assertColumnsAreEqual(endsResult, expectedEnds);
      assertColumnsAreEqual(expectedContains, containsResult);
      assertColumnsAreEqual(startsEmpty, expectedEmpty);
      assertColumnsAreEqual(endsEmpty, expectedEmpty);
      assertColumnsAreEqual(expectedEmpty, containsEmpty);
    }
  }

    @Test
    void testExtractRe() {
        try (ColumnVector input = ColumnVector.fromStrings("a1", "b2", "c3", null);
             Table expected = new Table.TestBuilder()
                     .column("a", "b", null, null)
                     .column("1", "2", null, null)
                     .build();
             Table found = input.extractRe("([ab])(\\d)")) {
            assertTablesAreEqual(expected, found);
        }
    }

  @Test
  void testMatchesRe() {
    String patternString1 = "\\d+";
    String patternString2 = "[A-Za-z]+\\s@[A-Za-z]+";
    String patternString3 = ".*";
    String patternString4 = "";
    try (ColumnVector testStrings = ColumnVector.fromStrings("", null, "abCD", "ovér the",
          "lazy @dog", "1234", "00:0:00");
         ColumnVector res1 = testStrings.matchesRe(patternString1);
         ColumnVector res2 = testStrings.matchesRe(patternString2);
         ColumnVector res3 = testStrings.matchesRe(patternString3);
         ColumnVector expected1 = ColumnVector.fromBoxedBooleans(false, null, false, false, false,
           true, true);
         ColumnVector expected2 = ColumnVector.fromBoxedBooleans(false, null, false, false, true,
           false, false);
         ColumnVector expected3 = ColumnVector.fromBoxedBooleans(true, null, true, true, true,
           true, true)) {
      assertColumnsAreEqual(expected1, res1);
      assertColumnsAreEqual(expected2, res2);
      assertColumnsAreEqual(expected3, res3);
    }
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector testStrings = ColumnVector.fromStrings("", null, "abCD", "ovér the",
             "lazy @dog", "1234", "00:0:00");
           ColumnVector res = testStrings.matchesRe(patternString4)) {}
    });
  }

  @Test
  void testContainsRe() {
    String patternString1 = "\\d+";
    String patternString2 = "[A-Za-z]+\\s@[A-Za-z]+";
    String patternString3 = ".*";
    String patternString4 = "";
    try (ColumnVector testStrings = ColumnVector.fromStrings(null, "abCD", "ovér the",
        "lazy @dog", "1234", "00:0:00", "abc1234abc", "there @are 2 lazy @dogs");
         ColumnVector res1 = testStrings.containsRe(patternString1);
         ColumnVector res2 = testStrings.containsRe(patternString2);
         ColumnVector res3 = testStrings.containsRe(patternString3);
         ColumnVector expected1 = ColumnVector.fromBoxedBooleans(null, false, false, false,
             true, true, true, true);
         ColumnVector expected2 = ColumnVector.fromBoxedBooleans(null, false, false, true,
             false, false, false, true);
         ColumnVector expected3 = ColumnVector.fromBoxedBooleans(null, true, true, true,
             true, true, true, true)) {
      assertColumnsAreEqual(expected1, res1);
      assertColumnsAreEqual(expected2, res2);
      assertColumnsAreEqual(expected3, res3);
    }
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector testStrings = ColumnVector.fromStrings("", null, "abCD", "ovér the",
          "lazy @dog", "1234", "00:0:00", "abc1234abc", "there @are 2 lazy @dogs");
           ColumnVector res = testStrings.containsRe(patternString4)) {}
    });
  }

  @Test
  @Disabled("Needs fix for https://github.com/rapidsai/cudf/issues/4671")
  void testContainsReEmptyInput() {
    String patternString1 = ".*";
    try (ColumnVector testStrings = ColumnVector.fromStrings("");
         ColumnVector res1 = testStrings.containsRe(patternString1);
         ColumnVector expected1 = ColumnVector.fromBoxedBooleans(true)) {
      assertColumnsAreEqual(expected1, res1);
    }
  }

  @Test
  void testStringFindOperationsThrowsException() {
    assertThrows(CudfException.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString(null);
           ColumnVector concat = sv.startsWith(emptyString)) {}
    });
    assertThrows(CudfException.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString(null);
           ColumnVector concat = sv.endsWith(emptyString)) {}
    });
    assertThrows(CudfException.class, () -> {
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar emptyString = Scalar.fromString(null);
           ColumnVector concat = sv.stringContains(emptyString)) {}
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
      try (ColumnVector sv = ColumnVector.fromStrings("a", "B", "cd");
           Scalar intScalar = Scalar.fromInt(1);
           ColumnVector concat = sv.stringContains(intScalar)) {}
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
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector v = ColumnVector.fromInts(1, 43, 42, 11, 2);
           Scalar patternString = Scalar.fromString("a");
           ColumnVector concat = v.stringContains(patternString)) {}
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
    assertThrows(CudfException.class, () -> {
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

  @Test
  void testsubstring() {
    try (ColumnVector v = ColumnVector.fromStrings("Héllo", "thésé", null,"", "ARé", "strings");
         ColumnVector e_allParameters = ColumnVector.fromStrings("llo", "ésé", null, "", "é", "rin");
         ColumnVector e_withoutStop = ColumnVector.fromStrings("llo", "ésé", null, "", "é", "rings");
         ColumnVector substring_allParam = v.substring(2, 5);
         ColumnVector substring_NoEnd = v.substring(2)) {
      assertColumnsAreEqual(e_allParameters, substring_allParam);
      assertColumnsAreEqual(e_withoutStop, substring_NoEnd);
    }
  }

  @Test
  void teststringSplit() {
    try (ColumnVector v = ColumnVector.fromStrings("Héllo there", "thésé", null, "", "ARé some", "test strings");
         Table expected = new Table.TestBuilder().column("Héllo", "thésé", null, "", "ARé", "test")
         .column("there", null, null, null, "some", "strings")
         .build();
         Scalar pattern = Scalar.fromString(" ");
         Table result = v.stringSplit(pattern)) {
      assertTablesAreEqual(expected, result);
    }
  }

  @Test
  void teststringSplitWhiteSpace() {
    try (ColumnVector v = ColumnVector.fromStrings("Héllo thesé", null, "are\tsome", "tést\nString", " ");
         Table expected = new Table.TestBuilder().column("Héllo", null, "are", "tést", null)
         .column("thesé", null, "some", "String", null)
         .build();
         Table result = v.stringSplit()) {
      assertTablesAreEqual(expected, result);
    }
  }

  @Test
  void teststringSplitThrowsException() {
    assertThrows(CudfException.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("Héllo", "thésé", null, "", "ARé", "strings");
           Scalar delimiter = Scalar.fromString(null);
           Table result = cv.stringSplit(delimiter)) {}
    });
    assertThrows(AssertionError.class, () -> {
    try (ColumnVector cv = ColumnVector.fromStrings("Héllo", "thésé", null, "", "ARé", "strings");
         Scalar delimiter = Scalar.fromInt(1);
         Table result = cv.stringSplit(delimiter)) {}
    });
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector cv = ColumnVector.fromStrings("Héllo", "thésé", null, "", "ARé", "strings");
           Table result = cv.stringSplit(null)) {}
    });
  }

  @Test
  void testsubstringColumn() {
    try (ColumnVector v = ColumnVector.fromStrings("Héllo", "thésé", null, "", "ARé", "strings");
         ColumnVector start = ColumnVector.fromInts(2, 1, 1, 1, 0, 1);
         ColumnVector end = ColumnVector.fromInts(5, 3, 1, 1, -1, -1);
         ColumnVector expected = ColumnVector.fromStrings("llo", "hé", null, "", "ARé", "trings");
         ColumnVector result = v.substring(start, end)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testsubstringThrowsException() {
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector v = ColumnVector.fromStrings("Héllo", "thésé", null, "", "ARé", "strings");
           ColumnVector start = ColumnVector.fromInts(2, 1, 1, 1, 0, 1);
           ColumnVector end = ColumnVector.fromInts(5, 3, 1, 1, -1);
           ColumnVector substring = v.substring(start, end)) {
      }
    });
  }

  @Test
  void teststringReplace() {
    try (ColumnVector v = ColumnVector.fromStrings("Héllo", "thésssé", null, "", "ARé", "sssstrings");
         ColumnVector e_allParameters = ColumnVector.fromStrings("Héllo", "théSsé", null, "", "ARé", "SStrings");
         Scalar target = Scalar.fromString("ss");
         Scalar replace = Scalar.fromString("S");
         ColumnVector replace_allParameters = v.stringReplace(target, replace)) {
      assertColumnsAreEqual(e_allParameters, replace_allParameters);
    }
  }

  @Test
  void teststringReplaceThrowsException() {
    assertThrows(AssertionError.class, () -> {
      try (ColumnVector testStrings = ColumnVector.fromStrings("Héllo", "thésé", null, "", "ARé", "strings");
           Scalar target= Scalar.fromString("");
           Scalar replace=Scalar.fromString("a");
           ColumnVector result = testStrings.stringReplace(target,replace)){}
    });
  }

  @Test
  void testStringReplaceWithBackrefs() {

    try (ColumnVector v = ColumnVector.fromStrings("<h1>title</h1>", "<h1>another title</h1>",
        null);
         ColumnVector expected = ColumnVector.fromStrings("<h2>title</h2>",
             "<h2>another title</h2>", null);
         ColumnVector actual = v.stringReplaceWithBackrefs("<h1>(.*)</h1>", "<h2>\\1</h2>")) {
      assertColumnsAreEqual(expected, actual);
    }

    try (ColumnVector v = ColumnVector.fromStrings("2020-1-01", "2020-2-02", null);
         ColumnVector expected = ColumnVector.fromStrings("2020-01-01", "2020-02-02", null);
         ColumnVector actual = v.stringReplaceWithBackrefs("-([0-9])-", "-0\\1-")) {
      assertColumnsAreEqual(expected, actual);
    }

    try (ColumnVector v = ColumnVector.fromStrings("2020-01-1", "2020-02-2",
        "2020-03-3invalid", null);
         ColumnVector expected = ColumnVector.fromStrings("2020-01-01", "2020-02-02",
             "2020-03-3invalid", null);
         ColumnVector actual = v.stringReplaceWithBackrefs(
             "-([0-9])$", "-0\\1")) {
      assertColumnsAreEqual(expected, actual);
    }

    try (ColumnVector v = ColumnVector.fromStrings("2020-01-1 random_text", "2020-02-2T12:34:56",
        "2020-03-3invalid", null);
         ColumnVector expected = ColumnVector.fromStrings("2020-01-01 random_text",
             "2020-02-02T12:34:56", "2020-03-3invalid", null);
         ColumnVector actual = v.stringReplaceWithBackrefs(
             "-([0-9])([ T])", "-0\\1\\2")) {
      assertColumnsAreEqual(expected, actual);
    }

  }

  @Test
  void testZfill() {
    try (ColumnVector v = ColumnVector.fromStrings("1", "23", "45678", null);
         ColumnVector expected = ColumnVector.fromStrings("01", "23", "45678", null);
         ColumnVector actual = v.zfill(2)) {
      assertColumnsAreEqual(expected, actual);
    }
    try (ColumnVector v = ColumnVector.fromStrings("1", "23", "45678", null);
         ColumnVector expected = ColumnVector.fromStrings("0001", "0023", "45678", null);
         ColumnVector actual = v.zfill(4)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testStringTitlize() {
    try (ColumnVector cv = ColumnVector.fromStrings("sPark", "sqL", "lowercase", null, "", "UPPERCASE");
         ColumnVector result = cv.toTitle();
         ColumnVector expected = ColumnVector.fromStrings("Spark", "Sql", "Lowercase", null, "", "Uppercase")) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testNansToNulls() {
    Float[] floats = new Float[]{1.2f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, null,
        Float.NaN, Float.MAX_VALUE, Float.MIN_VALUE, 435243.2323f, POSITIVE_FLOAT_NAN_LOWER_RANGE,
        POSITIVE_FLOAT_NAN_UPPER_RANGE, NEGATIVE_FLOAT_NAN_LOWER_RANGE,
        NEGATIVE_FLOAT_NAN_UPPER_RANGE};

    Float[] expectedFloats = new Float[]{1.2f, Float.POSITIVE_INFINITY,
        Float.NEGATIVE_INFINITY, null, null, Float.MAX_VALUE, Float.MIN_VALUE, 435243.2323f,
        null, null, null, null};

    Double[] doubles = new Double[]{1.2d, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, null,
        Double.NaN, Double.MAX_VALUE, Double.MIN_VALUE, 435243.2323d, POSITIVE_DOUBLE_NAN_LOWER_RANGE,
        POSITIVE_DOUBLE_NAN_UPPER_RANGE, NEGATIVE_DOUBLE_NAN_LOWER_RANGE,
        NEGATIVE_DOUBLE_NAN_UPPER_RANGE};

   Double[] expectedDoubles = new Double[]{1.2d, Double.POSITIVE_INFINITY,
        Double.NEGATIVE_INFINITY, null, null, Double.MAX_VALUE, Double.MIN_VALUE,
        435243.2323d, null, null, null, null};

    try (ColumnVector cvFloat = ColumnVector.fromBoxedFloats(floats);
         ColumnVector cvDouble = ColumnVector.fromBoxedDoubles(doubles);
         ColumnVector resultFloat = cvFloat.nansToNulls();
         ColumnVector resultDouble = cvDouble.nansToNulls();
         ColumnVector expectedFloat = ColumnVector.fromBoxedFloats(expectedFloats);
         ColumnVector expectedDouble = ColumnVector.fromBoxedDoubles(expectedDoubles)) {
      assertColumnsAreEqual(expectedFloat, resultFloat);
      assertColumnsAreEqual(expectedDouble, resultDouble);
    }
  }

  @Test
  void testIsInteger() {
    String[] intStrings = {"A", "nan", "Inf", "-Inf", "Infinity", "infinity", "2147483647",
        "2147483648", "-2147483648", "-2147483649", "NULL", "null", null, "1.2", "1.2e-4", "0.00012"};
    String[] longStrings = {"A", "nan", "Inf", "-Inf", "Infinity", "infinity",
        "9223372036854775807", "9223372036854775808", "-9223372036854775808",
        "-9223372036854775809", "NULL", "null", null, "1.2", "1.2e-4", "0.00012"};
    try (ColumnVector intStringCV = ColumnVector.fromStrings(intStrings);
         ColumnVector longStringCV = ColumnVector.fromStrings(longStrings);
         ColumnVector isInt = intStringCV.isInteger();
         ColumnVector isLong = longStringCV.isInteger();
         ColumnVector ints = intStringCV.asInts();
         ColumnVector longs = longStringCV.asLongs();
         ColumnVector expectedInts = ColumnVector.fromBoxedInts(0, 0, 0, 0, 0, 0, Integer.MAX_VALUE,
             Integer.MIN_VALUE, Integer.MIN_VALUE, Integer.MAX_VALUE, 0, 0, null, 1, 1, 0);
         ColumnVector expectedLongs = ColumnVector.fromBoxedLongs(0l, 0l, 0l, 0l, 0l, 0l, Long.MAX_VALUE,
             Long.MIN_VALUE, Long.MIN_VALUE, Long.MAX_VALUE, 0l, 0l, null, 1l, 1l, 0l);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false, false,
             false, true, true, true, true, false, false, null, false, false, false)) {
      assertColumnsAreEqual(expected, isInt);
      assertColumnsAreEqual(expected, isLong);
      assertColumnsAreEqual(expectedInts, ints);
      assertColumnsAreEqual(expectedLongs, longs);
    }
  }

  @Test
  void testIsFloat() {
    String[] floatStrings = {"A", "nan", "Inf", "-Inf", "Infinity", "infinity", "-0.0", "0.0",
        "3.4028235E38", "3.4028236E38", "-3.4028235E38", "-3.4028236E38", "1.2e-24", "NULL", "null",
        null, "423"};
    String[] doubleStrings = {"A", "nan", "Inf", "-Inf", "Infinity", "infinity", "-0.0", "0.0",
        "1.7976931348623159E308", "1.7976931348623160E308", "-1.7976931348623159E308",
        "-1.7976931348623160E308", "1.2e-234", "NULL", "null", null, "423"};
    try (ColumnVector floatStringCV = ColumnVector.fromStrings(floatStrings);
         ColumnVector doubleStringCV = ColumnVector.fromStrings(doubleStrings);
         ColumnVector isFloat = floatStringCV.isFloat();
         ColumnVector isDouble = doubleStringCV.isFloat();
         ColumnVector doubles = doubleStringCV.asDoubles();
         ColumnVector floats = floatStringCV.asFloats();
         ColumnVector expectedFloats = ColumnVector.fromBoxedFloats(0f, 0f, Float.POSITIVE_INFINITY,
             Float.NEGATIVE_INFINITY, 0f, 0f, -0f, 0f, Float.MAX_VALUE, Float.POSITIVE_INFINITY,
             -Float.MAX_VALUE, Float.NEGATIVE_INFINITY, 1.2e-24f, 0f, 0f, null, 423f);
         ColumnVector expectedDoubles = ColumnVector.fromBoxedDoubles(0d, 0d,
             Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 0d, 0d, -0d, 0d, Double.MAX_VALUE,
             Double.POSITIVE_INFINITY, -Double.MAX_VALUE, Double.NEGATIVE_INFINITY, 1.2e-234d, 0d,
             0d, null, 423d);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, true, false,
             false, true, true, true, true, true, true, true, false, false, null, true)) {
      assertColumnsAreEqual(expected, isFloat);
      assertColumnsAreEqual(expected, isDouble);
      assertColumnsAreEqual(expectedFloats, floats);
      assertColumnsAreEqual(expectedDoubles, doubles);
    }
  }
}
