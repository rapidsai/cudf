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

  interface CpuOpVV {
    void computeNullSafe(ColumnVector.Builder ret, ColumnVector lhs, ColumnVector rhs, int index);
  }

  interface CpuOpVS<S> {
    void computeNullSafe(ColumnVector.Builder ret, ColumnVector lhs, S rhs, int index);
  }

  interface CpuOpSV<S> {
    void computeNullSafe(ColumnVector.Builder ret, S lhs, ColumnVector rhs, int index);
  }

  public static ColumnVector forEach(DType retType, ColumnVector lhs, ColumnVector rhs, CpuOpVV op) {
    int len = (int)lhs.getRowCount();
    try (ColumnVector.Builder builder = ColumnVector.builder(retType, len)) {
      for (int i = 0; i < len; i++) {
        if (lhs.isNull(i) || rhs.isNull(i)) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, lhs, rhs, i);
        }
      }
      return builder.build();
    }
  }

  public static <S> ColumnVector forEachS(DType retType, ColumnVector lhs, S rhs, CpuOpVS<S> op) {
    int len = (int)lhs.getRowCount();
    try (ColumnVector.Builder builder = ColumnVector.builder(retType, len)) {
      for (int i = 0; i < len; i++) {
        if (lhs.isNull(i) || rhs == null) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, lhs, rhs, i);
        }
      }
      return builder.build();
    }
  }

  public static <S> ColumnVector forEachS(DType retType, S lhs, ColumnVector rhs, CpuOpSV<S> op) {
    int len = (int)rhs.getRowCount();
    try (ColumnVector.Builder builder = ColumnVector.builder(retType, len)) {
      for (int i = 0; i < len; i++) {
        if (rhs.isNull(i) || lhs == null) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, lhs, rhs, i);
        }
      }
      return builder.build();
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

      try (ColumnVector add = lcv1.add(Scalar.fromFloat(1.1f));
           ColumnVector expected = forEachS(DType.FLOAT32, lcv1, 1.1f,
                   (b, l, r, i) -> b.append(l.getLong(i) + r))) {
        assertColumnsAreEqual(expected, add, "int64 + scalar float");
      }

      try (ColumnVector add = Scalar.fromShort((short) 100).add(bcv1);
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

      try (ColumnVector sub = lcv1.sub(Scalar.fromFloat(1.1f));
           ColumnVector expected = forEachS(DType.FLOAT32, lcv1, 1.1f,
                   (b, l, r, i) -> b.append(l.getLong(i) - r))) {
        assertColumnsAreEqual(expected, sub, "int64 - scalar float");
      }

      try (ColumnVector sub = Scalar.fromShort((short) 100).sub(bcv1);
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

      try (ColumnVector answer = icv.mul(Scalar.fromFloat(1.1f));
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) * r))) {
        assertColumnsAreEqual(expected, answer, "int64 * scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).mul(icv);
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

      try (ColumnVector answer = icv.div(Scalar.fromFloat(1.1f));
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) / r))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).div(icv);
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

      try (ColumnVector answer = icv.trueDiv(Scalar.fromFloat(1.1f));
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) / r))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).trueDiv(icv);
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

      try (ColumnVector answer = icv.floorDiv(Scalar.fromFloat(1.1f));
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append((float)Math.floor(l.getInt(i) / r)))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).floorDiv(icv);
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

      try (ColumnVector answer = icv.mod(Scalar.fromFloat(1.1f));
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) % r))) {
        assertColumnsAreEqual(expected, answer, "int64 % scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).mod(icv);
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

      try (ColumnVector answer = icv.pow(Scalar.fromFloat(1.1f));
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append((float)Math.pow(l.getInt(i), r)))) {
        assertColumnsAreEqual(expected, answer, "int64 pow scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).pow(icv);
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

      try (ColumnVector answer = icv.equalTo(Scalar.fromFloat(1.0f));
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) == r))) {
        assertColumnsAreEqual(expected, answer, "int64 == scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).equalTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l == r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short == int32");
      }
    }
  }

  @Test
  public void testStringCategoryEqualScalar() {
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
  public void testStringCategoryEqualScalarNotPresent() {
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("boo");

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

      try (ColumnVector answer = icv.notEqualTo(Scalar.fromFloat(1.0f));
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) != r))) {
        assertColumnsAreEqual(expected, answer, "int64 != scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).notEqualTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l != r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short != int32");
      }
    }
  }

  @Test
  public void testStringCategoryNotEqualScalar() {
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
  public void testStringCategoryNotEqualScalarNotPresent() {
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("abc");

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

      try (ColumnVector answer = icv.lessThan(Scalar.fromFloat(1.0f));
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) < r))) {
        assertColumnsAreEqual(expected, answer, "int64 < scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).lessThan(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l < r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short < int32");
      }
    }
  }

  @Test
  public void testStringCategoryLessThanScalar() {
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
  public void testStringCategoryLessThanScalarNotPresent() {
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("abc");

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

      try (ColumnVector answer = icv.greaterThan(Scalar.fromFloat(1.0f));
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) > r))) {
        assertColumnsAreEqual(expected, answer, "int64 > scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).greaterThan(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l > r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short > int32");
      }
    }
  }

  @Test
  public void testStringCategoryGreaterThanScalar() {
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
  public void testStringCategoryGreaterThanScalarNotPresent() {
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("boo");

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

      try (ColumnVector answer = icv.lessOrEqualTo(Scalar.fromFloat(1.0f));
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) <= r))) {
        assertColumnsAreEqual(expected, answer, "int64 <= scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).lessOrEqualTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l <= r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short <= int32");
      }
    }
  }

  @Test
  public void testStringCategoryLessOrEqualToScalar() {
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
  public void testStringCategoryLessOrEqualToScalarNotPresent() {
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("boo");

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

      try (ColumnVector answer = icv.greaterOrEqualTo(Scalar.fromFloat(1.0f));
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) >= r))) {
      assertColumnsAreEqual(expected, answer, "int64 >= scalar float");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).greaterOrEqualTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l >= r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short >= int32");
      }
    }
  }

  @Test
  public void testStringCategoryGreaterOrEqualToScalar() {
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

  @Test
  public void testStringCategoryGreaterOrEqualToScalarNotPresent() {
    try (ColumnVector a = ColumnVector.categoryFromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.categoryFromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.categoryFromStrings("a", null, "b", null)) {
      Scalar s = Scalar.fromString("abc");

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

      try (ColumnVector answer = icv1.bitAnd(Scalar.fromInt(0x01));
           ColumnVector expected = forEachS(DType.INT32, icv1, 0x01,
                   (b, l, r, i) -> b.append(l.getInt(i) & r))) {
        assertColumnsAreEqual(expected, answer, "int32 & scalar int32");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).bitAnd(icv1);
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

      try (ColumnVector answer = icv1.bitOr(Scalar.fromInt(0x01));
           ColumnVector expected = forEachS(DType.INT32, icv1, 0x01,
                   (b, l, r, i) -> b.append(l.getInt(i) | r))) {
        assertColumnsAreEqual(expected, answer, "int32 | scalar int32");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).bitOr(icv1);
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

      try (ColumnVector answer = icv1.bitXor(Scalar.fromInt(0x01));
           ColumnVector expected = forEachS(DType.INT32, icv1, 0x01,
                   (b, l, r, i) -> b.append(l.getInt(i) ^ r))) {
        assertColumnsAreEqual(expected, answer, "int32 ^ scalar int32");
      }

      try (ColumnVector answer = Scalar.fromShort((short) 100).bitXor(icv1);
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

      try (ColumnVector answer = icv1.and(Scalar.fromBool(true));
           ColumnVector expected = forEachS(DType.BOOL8, icv1, true,
               (b, l, r, i) -> b.append(l.getBoolean(i) && r))) {
        assertColumnsAreEqual(expected, answer, "boolean AND true");
      }

      try (ColumnVector answer = icv1.and(Scalar.fromBool(false));
           ColumnVector expected = forEachS(DType.BOOL8, icv1, false,
                   (b, l, r, i) -> b.append(l.getBoolean(i) && r))) {
        assertColumnsAreEqual(expected, answer, "boolean AND false");
      }

      try (ColumnVector answer = icv1.and(Scalar.fromBool(true));
           ColumnVector expected = forEachS(DType.BOOL8, true, icv1,
               (b, l, r, i) -> b.append(l && r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "true AND boolean");
      }

      try (ColumnVector answer = icv1.and(Scalar.fromBool(false));
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

      try (ColumnVector answer = icv1.or(Scalar.fromBool(true));
           ColumnVector expected = forEachS(DType.BOOL8, icv1, true,
                   (b, l, r, i) -> b.append(l.getBoolean(i) || r))) {
        assertColumnsAreEqual(expected, answer, "boolean OR true");
      }

      try (ColumnVector answer = icv1.or(Scalar.fromBool(false));
           ColumnVector expected = forEachS(DType.BOOL8, icv1, false,
               (b, l, r, i) -> b.append(l.getBoolean(i) || r))) {
        assertColumnsAreEqual(expected, answer, "boolean OR false");
      }

      try (ColumnVector answer = icv1.or(Scalar.fromBool(true));
           ColumnVector expected = forEachS(DType.BOOL8, true, icv1,
                   (b, l, r, i) -> b.append(l || r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "true OR boolean");
      }

      try (ColumnVector answer = icv1.or(Scalar.fromBool(false));
           ColumnVector expected = forEachS(DType.BOOL8, false, icv1,
               (b, l, r, i) -> b.append(l || r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "false OR boolean");
      }
    }
  }
}
