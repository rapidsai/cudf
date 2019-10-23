
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

public class UnaryOpTest extends CudfTestBase {
  private static final Double[] DOUBLES_1 = new Double[]{1.0, 10.0, -100.1, 5.3, 50.0, 100.0, null};
  private static final Integer[] INTS_1 = new Integer[]{1, 10, -100, 5, 50, 100, null};
  private static final Boolean[] BOOLEANS_1 = new Boolean[]{true, false, true, false, true, false, null};
  private static final String[] STRINGS_1 = new String[]{"1", "10", "-100", "5", "50", "100", null};

  interface CpuOp {
    void computeNullSafe(ColumnVector.Builder ret, ColumnVector input, int index);
  }

  interface DoubleFun {
    double apply(double val);
  }

  static DoubleCpuOp doubleFun(DoubleFun fun) {
    return new DoubleCpuOp(fun);
  }

  static class DoubleCpuOp implements CpuOp {
    private final DoubleFun fun;

    DoubleCpuOp(DoubleFun fun) {
      this.fun = fun;
    }

    @Override
    public void computeNullSafe(ColumnVector.Builder ret, ColumnVector input, int index) {
      ret.append(fun.apply(input.getDouble(index)));
    }
  }

  interface IntFun {
    int apply(int val);
  }

  static IntCpuOp intFun(IntFun fun) {
    return new IntCpuOp(fun);
  }

  static class IntCpuOp implements CpuOp {
    private final IntFun fun;

    IntCpuOp(IntFun fun) {
      this.fun = fun;
    }

    @Override
    public void computeNullSafe(ColumnVector.Builder ret, ColumnVector input, int index) {
      ret.append(fun.apply(input.getInt(index)));
    }
  }

  interface BoolFun {
    boolean apply(boolean val);
  }

  static BoolCpuOp boolFun(BoolFun fun) {
    return new BoolCpuOp(fun);
  }

  static class BoolCpuOp implements CpuOp {
    private final BoolFun fun;

    BoolCpuOp(BoolFun fun) {
      this.fun = fun;
    }

    @Override
    public void computeNullSafe(ColumnVector.Builder ret, ColumnVector input, int index) {
      ret.append(fun.apply(input.getBoolean(index)));
    }
  }

  public static ColumnVector forEach(ColumnVector input, CpuOp op) {
    int len = (int)input.getRowCount();
    try (ColumnVector.Builder builder = ColumnVector.builder(input.getType(), len)) {
      for (int i = 0; i < len; i++) {
        if (input.isNull(i)) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, input, i);
        }
      }
      return builder.build();
    }
  }

  // These tests are not for the correctness of the underlying implementation, but really just
  // plumbing

  @Test
  public void testSin() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.sin();
         ColumnVector expected = forEach(dcv, doubleFun(Math::sin))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testCos() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.cos();
         ColumnVector expected = forEach(dcv, doubleFun(Math::cos))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testTan() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.tan();
         ColumnVector expected = forEach(dcv, doubleFun(Math::tan))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testArcsin() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.arcsin();
         ColumnVector expected = forEach(dcv, doubleFun(Math::asin))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testArccos() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.arccos();
         ColumnVector expected = forEach(dcv, doubleFun(Math::acos))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testArctan() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.arctan();
         ColumnVector expected = forEach(dcv, doubleFun(Math::atan))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testExp() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.exp();
         ColumnVector expected = forEach(dcv, doubleFun(Math::exp))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testLog() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.log();
         ColumnVector expected = forEach(dcv, doubleFun(Math::log))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testSqrt() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.sqrt();
         ColumnVector expected = forEach(dcv, doubleFun(Math::sqrt))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testCeil() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.ceil();
         ColumnVector expected = forEach(dcv, doubleFun(Math::ceil))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testFloor() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.floor();
         ColumnVector expected = forEach(dcv, doubleFun(Math::floor))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testAbs() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.abs();
         ColumnVector expected = forEach(dcv, doubleFun(Math::abs))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testBitInvert() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector answer = icv.bitInvert();
         ColumnVector expected = forEach(icv, intFun((i) -> ~i))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testNot() {
    try (ColumnVector icv = ColumnVector.fromBoxedBooleans(BOOLEANS_1);
         ColumnVector answer = icv.not();
         ColumnVector expected = forEach(icv, boolFun((i) -> !i))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  // String to string cat conversion has more to do with correctness as we wrote that all ourselves
  @Test
  public void testStringCastFullCircle() {
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
