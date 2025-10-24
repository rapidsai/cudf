
/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.Builder;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class UnaryOpTest extends CudfTestBase {
  private static final Double[] DOUBLES_1 = new Double[]{1.0, 10.0, -100.1, 5.3, 50.0, 100.0, null, Double.NaN, Double.POSITIVE_INFINITY, 1/9.0, Double.NEGATIVE_INFINITY, 500.0, -500.0};
  private static final Integer[] INTS_1 = new Integer[]{1, 10, -100, 5, 50, 100, null};
  private static final Boolean[] BOOLEANS_1 = new Boolean[]{true, false, true, false, true, false, null};

  interface CpuOp {
    void computeNullSafe(Builder ret, HostColumnVector input, int index);
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
    public void computeNullSafe(Builder ret, HostColumnVector input, int index) {
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
    public void computeNullSafe(Builder ret, HostColumnVector input, int index) {
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
    public void computeNullSafe(Builder ret, HostColumnVector input, int index) {
      ret.append(fun.apply(input.getBoolean(index)));
    }
  }

  public static ColumnVector forEach(ColumnVector input, CpuOp op) {
    int len = (int)input.getRowCount();
    try (HostColumnVector host = input.copyToHost();
         Builder builder = HostColumnVector.builder(input.getType(), len)) {
      for (int i = 0; i < len; i++) {
        if (host.isNull(i)) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, host, i);
        }
      }
      return builder.buildAndPutOnDevice();
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
  public void testSinh() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.sinh();
         ColumnVector expected = forEach(dcv, doubleFun(Math::sinh))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testCosh() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.cosh();
         ColumnVector expected = forEach(dcv, doubleFun(Math::cosh))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testTanh() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.tanh();
         ColumnVector expected = forEach(dcv, doubleFun(Math::tanh))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  public static double asinh(double value) {
    return value == Double.NEGATIVE_INFINITY ? Double.NEGATIVE_INFINITY :
      java.lang.StrictMath.log(value + java.lang.Math.sqrt(value * value + 1.0));
  }

  @Test
  public void testArcsinh() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.arcsinh();
         ColumnVector expected = forEach(dcv, doubleFun(UnaryOpTest::asinh))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  public static double acosh(double value) {
    return java.lang.StrictMath.log(value + java.lang.Math.sqrt(value * value - 1.0));
  }

  @Test
  public void testArccosh() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.arccosh();
         ColumnVector expected = forEach(dcv, doubleFun(UnaryOpTest::acosh))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  public static double atanh(double value) {
    return 0.5 * (java.lang.StrictMath.log1p(value) - java.lang.StrictMath.log1p(- value));
  }

  @Test
  public void testArctanh() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.arctanh();
         ColumnVector expected = forEach(dcv, doubleFun(UnaryOpTest::atanh))) {
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
  public void testLog2() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.log2();
         ColumnVector expected = forEach(dcv, doubleFun(n -> Math.log(n) / Math.log(2)))) {
      assertColumnsAreEqual(expected, answer);
    }
  }

  @Test
  public void testLog10() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.log10();
         ColumnVector expected = forEach(dcv, doubleFun(Math::log10))) {
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
  public void testCbrt() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.cbrt();
         ColumnVector expected = forEach(dcv, doubleFun(Math::cbrt))) {
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
  public void testRint() {
    try (ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector answer = dcv.rint();
         ColumnVector expected = forEach(dcv, doubleFun(Math::rint))) {
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
}
