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

import ai.rapids.cudf.HostColumnVector.Builder;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.IntStream;

import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;

public class BinaryOpTest extends CudfTestBase {
  private static final Integer[] INTS_1 = new Integer[]{1, 2, 3, 4, 5, null, 100};
  private static final Integer[] INTS_2 = new Integer[]{10, 20, 30, 40, 50, 60, 100};
  private static final Byte[] BYTES_1 = new Byte[]{-1, 7, 123, null, 50, 60, 100};
  private static final Float[] FLOATS_1 = new Float[]{1f, 10f, 100f, 5.3f, 50f, 100f, null};
  private static final Float[] FLOATS_2 = new Float[]{10f, 20f, 30f, 40f, 50f, 60f, 100f};
  private static final Long[] LONGS_1 = new Long[]{1L, 2L, 3L, 4L, 5L, null, 100L};
  private static final Long[] LONGS_2 = new Long[]{10L, 20L, 30L, 40L, 50L, 60L, 100L};
  private static final Double[] DOUBLES_1 = new Double[]{1.0, 10.0, 100.0, 5.3, 50.0, 100.0, null};
  private static final Double[] DOUBLES_2 = new Double[]{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 100.0};
  private static final Boolean[] BOOLEANS_1 = new Boolean[]{true, true, false, false, null};
  private static final Boolean[] BOOLEANS_2 = new Boolean[]{true, false, true, false, true};
  private static final int[] SHIFT_BY = new int[]{1, 2, 3, 4, 5, 10, 20};

  interface CpuOpVV {
    void computeNullSafe(Builder ret, HostColumnVector lhs, HostColumnVector rhs, int index);
  }

  interface CpuOpVS<S> {
    void computeNullSafe(Builder ret, HostColumnVector lhs, S rhs, int index);
  }

  interface CpuOpSV<S> {
    void computeNullSafe(Builder ret, S lhs, HostColumnVector rhs, int index);
  }

  public static ColumnVector forEach(DType retType, ColumnVector lhs, ColumnVector rhs, CpuOpVV op) {
    int len = (int)lhs.getRowCount();
    try (HostColumnVector hostLHS  = lhs.copyToHost();
         HostColumnVector hostRHS = rhs.copyToHost();
         Builder builder = HostColumnVector.builder(retType, len)) {
      for (int i = 0; i < len; i++) {
        if (hostLHS.isNull(i) || hostRHS.isNull(i)) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, hostLHS, hostRHS, i);
        }
      }
      return builder.buildAndPutOnDevice();
    }
  }

  public static <S> ColumnVector forEachS(DType retType, ColumnVector lhs, S rhs, CpuOpVS<S> op) {
    int len = (int)lhs.getRowCount();
    try (HostColumnVector hostLHS = lhs.copyToHost();
         Builder builder = HostColumnVector.builder(retType, len)) {
      for (int i = 0; i < len; i++) {
        if (hostLHS.isNull(i) || rhs == null) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, hostLHS, rhs, i);
        }
      }
      return builder.buildAndPutOnDevice();
    }
  }

  public static <S> ColumnVector forEachS(DType retType, S lhs, ColumnVector rhs, CpuOpSV<S> op) {
    int len = (int)rhs.getRowCount();
    try (HostColumnVector hostRHS = rhs.copyToHost();
        Builder builder = HostColumnVector.builder(retType, len)) {
      for (int i = 0; i < len; i++) {
        if (hostRHS.isNull(i) || lhs == null) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, lhs, hostRHS, i);
        }
      }
      return builder.buildAndPutOnDevice();
    }
  }

  @Test
  public void testAdd() {
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
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) + r.getInt(i)))) {
        assertColumnsAreEqual(expected, add, "int32");
      }

      try (ColumnVector add = icv1.add(bcv1);
           ColumnVector expected = forEach(DType.INT32, icv1, bcv1,
                   (b, l, r, i) -> b.append(l.getInt(i) + r.getByte(i)))) {
        assertColumnsAreEqual(expected, add, "int32 + byte");
      }

      try (ColumnVector add = fcv1.add(fcv2);
           ColumnVector expected = forEach(DType.FLOAT32, fcv1, fcv2,
                   (b, l, r, i) -> b.append(l.getFloat(i) + r.getFloat(i)))) {
        assertColumnsAreEqual(expected, add, "float32");
      }

      try (ColumnVector addIntFirst = icv1.add(fcv2, DType.FLOAT32);
           ColumnVector addFloatFirst = fcv2.add(icv1)) {
        assertColumnsAreEqual(addIntFirst, addFloatFirst, "int + float vs float + int");
      }

      try (ColumnVector add = lcv1.add(lcv2);
           ColumnVector expected = forEach(DType.INT64, lcv1, lcv2,
                   (b, l, r, i) -> b.append(l.getLong(i) + r.getLong(i)))) {
        assertColumnsAreEqual(expected, add, "int64");
      }

      try (ColumnVector add = lcv1.add(bcv1);
           ColumnVector expected = forEach(DType.INT64, lcv1, bcv1,
                   (b, l, r, i) -> b.append(l.getLong(i) + r.getByte(i)))) {
        assertColumnsAreEqual(expected, add, "int64 + byte");
      }

      try (ColumnVector add = dcv1.add(dcv2);
           ColumnVector expected = forEach(DType.FLOAT64, dcv1, dcv2,
                   (b, l, r, i) -> b.append(l.getDouble(i) + r.getDouble(i)))) {
        assertColumnsAreEqual(expected, add, "float64");
      }

      try (ColumnVector addIntFirst = icv1.add(dcv2, DType.FLOAT64);
           ColumnVector addDoubleFirst = dcv2.add(icv1)) {
        assertColumnsAreEqual(addIntFirst, addDoubleFirst, "int + double vs double + int");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector add = lcv1.add(s);
           ColumnVector expected = forEachS(DType.FLOAT32, lcv1, 1.1f,
                   (b, l, r, i) -> b.append(l.getLong(i) + r))) {
        assertColumnsAreEqual(expected, add, "int64 + scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector add = s.add(bcv1);
           ColumnVector expected = forEachS(DType.INT16, (short) 100,  bcv1,
                   (b, l, r, i) -> b.append((short)(l + r.getByte(i))))) {
        assertColumnsAreEqual(expected, add, "scalar short + byte");
      }
    }
  }

  @Test
  public void testSub() {
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
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) - r.getInt(i)))) {
        assertColumnsAreEqual(expected, sub, "int32");
      }

      try (ColumnVector sub = icv1.sub(bcv1);
           ColumnVector expected = forEach(DType.INT32, icv1, bcv1,
                   (b, l, r, i) -> b.append(l.getInt(i) - r.getByte(i)))) {
        assertColumnsAreEqual(expected, sub, "int32 - byte");
      }

      try (ColumnVector sub = fcv1.sub(fcv2);
           ColumnVector expected = forEach(DType.FLOAT32, fcv1, fcv2,
                   (b, l, r, i) -> b.append(l.getFloat(i) - r.getFloat(i)))) {
        assertColumnsAreEqual(expected, sub, "float32");
      }

      try (ColumnVector sub = icv1.sub(fcv2, DType.FLOAT32);
           ColumnVector expected = forEach(DType.FLOAT32, icv1, fcv2,
                   (b, l, r, i) -> b.append(l.getInt(i) - r.getFloat(i)))) {
        assertColumnsAreEqual(expected, sub, "int - float");
      }

      try (ColumnVector sub = lcv1.sub(lcv2);
           ColumnVector expected = forEach(DType.INT64, lcv1, lcv2,
                   (b, l, r, i) -> b.append(l.getLong(i) - r.getLong(i)))) {
        assertColumnsAreEqual(expected, sub, "int64");
      }

      try (ColumnVector sub = lcv1.sub(bcv1);
           ColumnVector expected = forEach(DType.INT64, lcv1, bcv1,
                   (b, l, r, i) -> b.append(l.getLong(i) - r.getByte(i)))) {
        assertColumnsAreEqual(expected, sub, "int64 - byte");
      }

      try (ColumnVector sub = dcv1.sub(dcv2);
           ColumnVector expected = forEach(DType.FLOAT64, dcv1, dcv2,
                   (b, l, r, i) -> b.append(l.getDouble(i) - r.getDouble(i)))) {
        assertColumnsAreEqual(expected, sub, "float64");
      }

      try (ColumnVector sub = dcv2.sub(icv1);
           ColumnVector expected = forEach(DType.FLOAT64, dcv2, icv1,
                   (b, l, r, i) -> b.append(l.getDouble(i) - r.getInt(i)))) {
        assertColumnsAreEqual(expected, sub, "double - int");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector sub = lcv1.sub(s);
           ColumnVector expected = forEachS(DType.FLOAT32, lcv1, 1.1f,
                   (b, l, r, i) -> b.append(l.getLong(i) - r))) {
        assertColumnsAreEqual(expected, sub, "int64 - scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector sub = s.sub(bcv1);
           ColumnVector expected = forEachS(DType.INT16, (short) 100,  bcv1,
                   (b, l, r, i) -> b.append((short)(l - r.getByte(i))))) {
        assertColumnsAreEqual(expected, sub, "scalar short - byte");
      }
    }
  }

  // The rest of the tests are very basic to ensure that operations plumbing is in place, not to
  // exhaustively test
  // The underlying implementation.

  @Test
  public void testMul() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.mul(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) * r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 * double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.mul(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) * r))) {
        assertColumnsAreEqual(expected, answer, "int64 * scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.mul(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l * r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short * int32");
      }
    }
  }

  @Test
  public void testDiv() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.div(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) / r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 / double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.div(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) / r))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.div(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l / r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short / int32");
      }
    }
  }

  @Test
  public void testTrueDiv() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.trueDiv(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) / r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 / double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.trueDiv(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) / r))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.trueDiv(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l / r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short / int32");
      }
    }
  }

  @Test
  public void testFloorDiv() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.floorDiv(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(Math.floor(l.getInt(i) / r.getDouble(i))))) {
        assertColumnsAreEqual(expected, answer, "int32 / double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.floorDiv(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append((float)Math.floor(l.getInt(i) / r)))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.floorDiv(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l / r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short / int32");
      }
    }
  }

  @Test
  public void testMod() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.mod(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) % r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 % double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.mod(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) % r))) {
        assertColumnsAreEqual(expected, answer, "int64 % scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.mod(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l % r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short % int32");
      }
    }
  }

  @Test
  public void testPow() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.pow(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(Math.pow(l.getInt(i), r.getDouble(i))))) {
        assertColumnsAreEqual(expected, answer, "int32 pow double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.pow(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append((float)Math.pow(l.getInt(i), r)))) {
        assertColumnsAreEqual(expected, answer, "int64 pow scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.pow(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append((int)Math.pow(l, r.getInt(i))))) {
        assertColumnsAreEqual(expected, answer, "scalar short pow int32");
      }
    }
  }

  @Test
  public void testEqual() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.equalTo(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) == r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 == double");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.equalTo(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) == r))) {
        assertColumnsAreEqual(expected, answer, "int64 == scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.equalTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l == r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short == int32");
      }
    }
  }

  @Test
  public void testStringEqualScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

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
  public void testStringEqualScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("boo")) {

      try (ColumnVector answer = a.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testNotEqual() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.notEqualTo(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) != r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 != double");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.notEqualTo(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) != r))) {
        assertColumnsAreEqual(expected, answer, "int64 != scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.notEqualTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l != r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short != int32");
      }
    }
  }

  @Test
  public void testStringNotEqualScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

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
  public void testStringNotEqualScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("abc")) {

      try (ColumnVector answer = a.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testLessThan() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.lessThan(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) < r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 < double");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.lessThan(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) < r))) {
        assertColumnsAreEqual(expected, answer, "int64 < scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.lessThan(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l < r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short < int32");
      }
    }
  }

  @Test
  public void testStringLessThanScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

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
  public void testStringLessThanScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("abc")) {

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
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.greaterThan(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) > r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 > double");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.greaterThan(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) > r))) {
        assertColumnsAreEqual(expected, answer, "int64 > scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.greaterThan(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l > r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short > int32");
      }
    }
  }

  @Test
  public void testStringGreaterThanScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

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
  public void testStringGreaterThanScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("boo")) {

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
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.lessOrEqualTo(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) <= r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 <= double");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.lessOrEqualTo(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) <= r))) {
        assertColumnsAreEqual(expected, answer, "int64 <= scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.lessOrEqualTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l <= r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short <= int32");
      }
    }
  }

  @Test
  public void testStringLessOrEqualToScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

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
  public void testStringLessOrEqualToScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("boo")) {

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
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.greaterOrEqualTo(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) >= r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 >= double");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.greaterOrEqualTo(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) >= r))) {
      assertColumnsAreEqual(expected, answer, "int64 >= scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.greaterOrEqualTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l >= r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short >= int32");
      }
    }
  }

  @Test
  public void testStringGreaterOrEqualToScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

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

  @Test
  public void testStringGreaterOrEqualToScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("abc")) {

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

  @Test
  public void testBitAnd() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2)) {
      try (ColumnVector answer = icv1.bitAnd(icv2);
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) & r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 & int32");
      }

      try (Scalar s = Scalar.fromInt(0x01);
           ColumnVector answer = icv1.bitAnd(s);
           ColumnVector expected = forEachS(DType.INT32, icv1, 0x01,
                   (b, l, r, i) -> b.append(l.getInt(i) & r))) {
        assertColumnsAreEqual(expected, answer, "int32 & scalar int32");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.bitAnd(icv1);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv1,
                   (b, l, r, i) -> b.append(l & r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short & int32");
      }
    }
  }

  @Test
  public void testBitOr() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2)) {
      try (ColumnVector answer = icv1.bitOr(icv2);
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) | r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 | int32");
      }

      try (Scalar s = Scalar.fromInt(0x01);
           ColumnVector answer = icv1.bitOr(s);
           ColumnVector expected = forEachS(DType.INT32, icv1, 0x01,
                   (b, l, r, i) -> b.append(l.getInt(i) | r))) {
        assertColumnsAreEqual(expected, answer, "int32 | scalar int32");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.bitOr(icv1);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv1,
                   (b, l, r, i) -> b.append(l | r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short | int32");
      }
    }
  }

  @Test
  public void testBitXor() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2)) {
      try (ColumnVector answer = icv1.bitXor(icv2);
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) ^ r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 ^ int32");
      }

      try (Scalar s = Scalar.fromInt(0x01);
           ColumnVector answer = icv1.bitXor(s);
           ColumnVector expected = forEachS(DType.INT32, icv1, 0x01,
                   (b, l, r, i) -> b.append(l.getInt(i) ^ r))) {
        assertColumnsAreEqual(expected, answer, "int32 ^ scalar int32");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.bitXor(icv1);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv1,
                   (b, l, r, i) -> b.append(l ^ r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short ^ int32");
      }
    }
  }

  @Test
  public void testAnd() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedBooleans(BOOLEANS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedBooleans(BOOLEANS_2)) {
      try (ColumnVector answer = icv1.and(icv2);
           ColumnVector expected = forEach(DType.BOOL8, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getBoolean(i) && r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "boolean AND boolean");
      }

      try (Scalar s = Scalar.fromBool(true);
           ColumnVector answer = icv1.and(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv1, true,
               (b, l, r, i) -> b.append(l.getBoolean(i) && r))) {
        assertColumnsAreEqual(expected, answer, "boolean AND true");
      }

      try (Scalar s = Scalar.fromBool(false);
           ColumnVector answer = icv1.and(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv1, false,
                   (b, l, r, i) -> b.append(l.getBoolean(i) && r))) {
        assertColumnsAreEqual(expected, answer, "boolean AND false");
      }

      try (Scalar s = Scalar.fromBool(true);
           ColumnVector answer = icv1.and(s);
           ColumnVector expected = forEachS(DType.BOOL8, true, icv1,
               (b, l, r, i) -> b.append(l && r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "true AND boolean");
      }

      try (Scalar s = Scalar.fromBool(false);
           ColumnVector answer = icv1.and(s);
           ColumnVector expected = forEachS(DType.BOOL8, false, icv1,
                   (b, l, r, i) -> b.append(l && r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "false AND boolean");
      }
    }
  }

  @Test
  public void testOr() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedBooleans(BOOLEANS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedBooleans(BOOLEANS_2)) {
      try (ColumnVector answer = icv1.or(icv2);
           ColumnVector expected = forEach(DType.BOOL8, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getBoolean(i) || r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "boolean OR boolean");
      }

      try (Scalar s = Scalar.fromBool(true);
           ColumnVector answer = icv1.or(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv1, true,
                   (b, l, r, i) -> b.append(l.getBoolean(i) || r))) {
        assertColumnsAreEqual(expected, answer, "boolean OR true");
      }

      try (Scalar s = Scalar.fromBool(false);
           ColumnVector answer = icv1.or(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv1, false,
               (b, l, r, i) -> b.append(l.getBoolean(i) || r))) {
        assertColumnsAreEqual(expected, answer, "boolean OR false");
      }

      try (Scalar s = Scalar.fromBool(true);
           ColumnVector answer = icv1.or(s);
           ColumnVector expected = forEachS(DType.BOOL8, true, icv1,
                   (b, l, r, i) -> b.append(l || r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "true OR boolean");
      }

      try (Scalar s = Scalar.fromBool(false);
           ColumnVector answer = icv1.or(s);
           ColumnVector expected = forEachS(DType.BOOL8, false, icv1,
               (b, l, r, i) -> b.append(l || r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "false OR boolean");
      }
    }
  }

  @Test
  public void testShiftLeft() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector shiftBy = ColumnVector.fromInts(SHIFT_BY)) {
      try (ColumnVector answer = icv.shiftLeft(shiftBy);
           ColumnVector expected = forEach(DType.INT32, icv, shiftBy,
               (b, l, r, i) -> b.append(l.getInt(i) << r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 shifted left");
      }

      try (Scalar s = Scalar.fromInt(4);
           ColumnVector answer = icv.shiftLeft(s, DType.INT64);
           ColumnVector expected = forEachS(DType.INT64, icv, 4,
               (b, l, r, i) -> b.append(((long)l.getInt(i) << r)))) {
        assertColumnsAreEqual(expected, answer, "int32 << scalar = int64");
      }

      try (Scalar s = Scalar.fromShort((short) 0x0000FFFF);
           ColumnVector answer = s.shiftLeft(shiftBy, DType.INT16);
           ColumnVector expected = forEachS(DType.INT16, (short) 0x0000FFFF,  shiftBy,
               (b, l, r, i) -> {
                 int shifted = l << r.getInt(i);
                 b.append((short) shifted);
               })) {
        assertColumnsAreEqual(expected, answer, "scalar short << int32");
      }
    }
  }

  @Test
  public void testShiftRight() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector shiftBy = ColumnVector.fromInts(SHIFT_BY)) {
      try (ColumnVector answer = icv.shiftRight(shiftBy);
           ColumnVector expected = forEach(DType.INT32, icv, shiftBy,
               (b, l, r, i) -> b.append(l.getInt(i) >> r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 shifted right");
      }

      try (Scalar s = Scalar.fromInt(4);
           ColumnVector answer = icv.shiftRight(s, DType.INT64);
           ColumnVector expected = forEachS(DType.INT64, icv, 4,
               (b, l, r, i) -> b.append(((long)(l.getInt(i) >> r))))) {
        assertColumnsAreEqual(expected, answer, "int32 >> scalar = int64");
      }

      try (Scalar s = Scalar.fromShort((short) 0x0000FFFF);
           ColumnVector answer = s.shiftRight(shiftBy, DType.INT16);
           ColumnVector expected = forEachS(DType.INT16, (short) 0x0000FFFF,  shiftBy,
               (b, l, r, i) -> {
                 int shifted = l >> r.getInt(i);
                 b.append((short) shifted);
               })) {
        assertColumnsAreEqual(expected, answer, "scalar short >> int32 = int16");
      }
    }
  }

  @Test
  public void testShiftRightUnsigned() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector shiftBy = ColumnVector.fromInts(SHIFT_BY)) {
      try (ColumnVector answer = icv.shiftRightUnsigned(shiftBy);
           ColumnVector expected = forEach(DType.INT32, icv, shiftBy,
               (b, l, r, i) -> b.append(l.getInt(i) >>> r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 shifted right unsigned");
      }

      try (Scalar s = Scalar.fromInt(4);
           ColumnVector answer = icv.shiftRightUnsigned(s, DType.INT64);
           ColumnVector expected = forEachS(DType.INT64, icv, 4,
               (b, l, r, i) -> b.append(((long)(l.getInt(i) >>> r))))) {
        assertColumnsAreEqual(expected, answer, "int32 >>> scalar = int64");
      }
    }
  }

  @Test
  public void testLogBase10() {
    try (ColumnVector dcv1 = ColumnVector.fromBoxedDoubles(DOUBLES_2);
         Scalar base = Scalar.fromInt(10);
         ColumnVector answer = dcv1.log(base);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Arrays.stream(DOUBLES_2)
            .map(Math::log10)
            .toArray(Double[]::new))) {
      assertColumnsAreEqual(expected, answer, "log10");
    }
  }

  @Test
  public void testLogBase2() {
    try (ColumnVector dcv1 = ColumnVector.fromBoxedDoubles(DOUBLES_2);
         Scalar base = Scalar.fromInt(2);
         ColumnVector answer = dcv1.log(base);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Arrays.stream(DOUBLES_2)
             .map(n -> Math.log(n) / Math.log(2))
             .toArray(Double[]::new))) {
      assertColumnsAreEqual(expected, answer, "log2");
    }
  }

  @Test
  public void testArctan2() {
    Double[] xValues = TestUtils.getDoubles(20342309423423L, 10, false);
    Double[] yValues = TestUtils.getDoubles(33244345234423L, 10, false);
    try (ColumnVector y = ColumnVector.fromBoxedDoubles(yValues);
         ColumnVector x = ColumnVector.fromBoxedDoubles(xValues);
         ColumnVector result = y.arctan2(x);
         ColumnVector expected = ColumnVector.fromDoubles(IntStream.range(0,xValues.length)
             .mapToDouble(n -> Math.atan2(yValues[n], xValues[n])).toArray())) {
      assertColumnsAreEqual(expected, result);
    }
  }
}
