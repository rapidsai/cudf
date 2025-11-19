/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.CudfTestBase;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.NullSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Stream;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class CompiledExpressionTest extends CudfTestBase {
  @Test
  public void testColumnReferenceTransform() {
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build()) {
      // use an implicit table reference
      ColumnReference expr = new ColumnReference(1);
      try (CompiledExpression compiledExpr = expr.compile();
           ColumnVector actual = compiledExpr.computeColumn(t)) {
        assertColumnsAreEqual(t.getColumn(1), actual);
      }

      // use an explicit table reference
      expr = new ColumnReference(1, TableReference.LEFT);
      try (CompiledExpression compiledExpr = expr.compile();
           ColumnVector actual = compiledExpr.computeColumn(t)) {
        assertColumnsAreEqual(t.getColumn(1), actual);
      }
    }
  }

  @Test
  public void testInvalidColumnReferenceTransform() {
    // Verify that computeColumn throws when passed an expression operating on TableReference.RIGHT.
    ColumnReference expr = new ColumnReference(1, TableReference.RIGHT);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile()) {
      Assertions.assertThrows(CudfException.class, () -> compiledExpr.computeColumn(t).close());
    }
  }

  @Test
  public void testBooleanLiteralTransform() {
    try (Table t = new Table.TestBuilder().column(true, false, null).build()) {
      Literal expr = Literal.ofBoolean(true);
      try (CompiledExpression trueCompiledExpr = expr.compile();
           ColumnVector trueExprActual = trueCompiledExpr.computeColumn(t);
           ColumnVector trueExprExpected = ColumnVector.fromBoxedBooleans(true, true, true)) {
        assertColumnsAreEqual(trueExprExpected, trueExprActual);
      }

      Literal nullLiteral = Literal.ofBoolean(null);
      UnaryOperation nullExpr = new UnaryOperation(UnaryOperator.IDENTITY, nullLiteral);
      try (CompiledExpression nullCompiledExpr = nullExpr.compile();
           ColumnVector nullExprActual = nullCompiledExpr.computeColumn(t);
           ColumnVector nullExprExpected = ColumnVector.fromBoxedBooleans(null, null, null)) {
        assertColumnsAreEqual(nullExprExpected, nullExprActual);
      }
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(bytes = 0x12)
  public void testByteLiteralTransform(Byte value) {
    Literal expr = Literal.ofByte(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBytes(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(shorts = 0x1234)
  public void testShortLiteralTransform(Short value) {
    Literal expr = Literal.ofShort(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedShorts(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(ints = 0x12345678)
  public void testIntLiteralTransform(Integer value) {
    Literal expr = Literal.ofInt(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedInts(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testLongLiteralTransform(Long value) {
    Literal expr = Literal.ofLong(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(floats = { 123456.789f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY} )
  public void testFloatLiteralTransform(Float value) {
    Literal expr = Literal.ofFloat(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedFloats(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(doubles = { 123456.789f, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY} )
  public void testDoubleLiteralTransform(Double value) {
    Literal expr = Literal.ofDouble(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(ints = 0x12345678)
  public void testTimestampDaysLiteralTransform(Integer value) {
    Literal expr = Literal.ofTimestampDaysFromInt(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampDaysFromBoxedInts(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testTimestampSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofTimestampFromLong(DType.TIMESTAMP_SECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testTimestampMilliSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofTimestampFromLong(DType.TIMESTAMP_MILLISECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampMilliSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testTimestampMicroSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofTimestampFromLong(DType.TIMESTAMP_MICROSECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampMicroSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testTimestampNanoSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofTimestampFromLong(DType.TIMESTAMP_NANOSECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampNanoSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(ints = 0x12345678)
  public void testDurationDaysLiteralTransform(Integer value) {
    Literal expr = Literal.ofDurationDaysFromInt(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationDaysFromBoxedInts(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testDurationSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofDurationFromLong(DType.DURATION_SECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testDurationMilliSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofDurationFromLong(DType.DURATION_MILLISECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationMilliSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testDurationMicroSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofDurationFromLong(DType.DURATION_MICROSECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationMicroSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testDurationNanoSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofDurationFromLong(DType.DURATION_NANOSECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationNanoSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static <T, R> ArrayList<R> mapArray(T[] input, Function<T, R> func) {
    ArrayList<R> result = new ArrayList<>(input.length);
    for (T t : input) {
      result.add(t == null ? null : func.apply(t));
    }
    return result;
  }

  private static <T, U, R> ArrayList<R> mapArray(T[] in1, U[] in2, BiFunction<T, U, R> func) {
    assert in1.length == in2.length;
    ArrayList<R> result = new ArrayList<>(in1.length);
    for (int i = 0; i < in1.length; i++) {
      result.add(in1[i] == null || in2[i] == null ? null : func.apply(in1[i], in2[i]));
    }
    return result;
  }

  private static Stream<Arguments> createUnaryDoubleOperationParams() {
    Double[] input = new Double[] { -5., 4.5, null, 2.7, 1.5 };
    return Stream.of(
        Arguments.of(UnaryOperator.IDENTITY, input, Arrays.asList(input)),
        Arguments.of(UnaryOperator.SIN, input, mapArray(input, Math::sin)),
        Arguments.of(UnaryOperator.COS, input, mapArray(input, Math::cos)),
        Arguments.of(UnaryOperator.TAN, input, mapArray(input, Math::tan)),
        Arguments.of(UnaryOperator.ARCSIN, input, mapArray(input, Math::asin)),
        Arguments.of(UnaryOperator.ARCCOS, input, mapArray(input, Math::acos)),
        Arguments.of(UnaryOperator.ARCTAN, input, mapArray(input, Math::atan)),
        Arguments.of(UnaryOperator.SINH, input, mapArray(input, Math::sinh)),
        Arguments.of(UnaryOperator.COSH, input, mapArray(input, Math::cosh)),
        Arguments.of(UnaryOperator.TANH, input, mapArray(input, Math::tanh)),
        Arguments.of(UnaryOperator.EXP, input, mapArray(input, Math::exp)),
        Arguments.of(UnaryOperator.LOG, input, mapArray(input, Math::log)),
        Arguments.of(UnaryOperator.SQRT, input, mapArray(input, Math::sqrt)),
        Arguments.of(UnaryOperator.CBRT, input, mapArray(input, Math::cbrt)),
        Arguments.of(UnaryOperator.CEIL, input, mapArray(input, Math::ceil)),
        Arguments.of(UnaryOperator.FLOOR, input, mapArray(input, Math::floor)),
        Arguments.of(UnaryOperator.ABS, input, mapArray(input, Math::abs)),
        Arguments.of(UnaryOperator.RINT, input, mapArray(input, Math::rint)));
  }

  @ParameterizedTest
  @MethodSource("createUnaryDoubleOperationParams")
  void testUnaryDoubleOperationTransform(UnaryOperator op, Double[] input,
                                          List<Double> expectedValues) {
    UnaryOperation expr = new UnaryOperation(op, new ColumnReference(0));
    try (Table t = new Table.TestBuilder().column(input).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(
             expectedValues.toArray(new Double[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testUnaryShortOperationTransform() {
    Short[] input = new Short[] { -5, 4, null, 2, 1 };
    try (Table t = new Table.TestBuilder().column(input).build()) {
      ColumnReference expr = new ColumnReference(0);
      try (CompiledExpression compiledExpr = expr.compile();
           ColumnVector actual = compiledExpr.computeColumn(t)) {
        assertColumnsAreEqual(t.getColumn(0), actual);
      }

      UnaryOperation expr2 = new UnaryOperation(UnaryOperator.BIT_INVERT, new ColumnReference(0));
      try (CompiledExpression compiledExpr = expr2.compile();
           ColumnVector actual = compiledExpr.computeColumn(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(4, -5, null, -3, -2)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  private static Stream<Arguments> createUnaryLogicalOperationParams() {
    Long[] input = new Long[] { -5L, 0L, null, 2L, 1L };
    return Stream.of(
        Arguments.of(UnaryOperator.NOT, input, Arrays.asList(false, true, null, false, false)),
        Arguments.of(UnaryOperator.IS_NULL, input, Arrays.asList(false, false, true, false, false)));
  }

  @ParameterizedTest
  @MethodSource("createUnaryLogicalOperationParams")
  void testUnaryLogicalOperationTransform(UnaryOperator op, Long[] input,
                                          List<Boolean> expectedValues) {
    UnaryOperation expr = new UnaryOperation(op, new ColumnReference(0));
    try (Table t = new Table.TestBuilder().column(input).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryFloatOperationParams() {
    Float[] in1 = new Float[] { -5f, 4.5f, null, 2.7f };
    Float[] in2 = new Float[] { 123f, -456f, null, 0f };
    return Stream.of(
        Arguments.of(BinaryOperator.ADD, in1, in2, mapArray(in1, in2, Float::sum)),
        Arguments.of(BinaryOperator.SUB, in1, in2, mapArray(in1, in2, (a, b) -> a - b)),
        Arguments.of(BinaryOperator.MUL, in1, in2, mapArray(in1, in2, (a, b) -> a * b)),
        Arguments.of(BinaryOperator.DIV, in1, in2, mapArray(in1, in2, (a, b) -> a / b)),
        Arguments.of(BinaryOperator.MOD, in1, in2, mapArray(in1, in2, (a, b) -> a % b)),
        Arguments.of(BinaryOperator.PYMOD, in1, in2, mapArray(in1, in2,
            (a, b) -> ((a % b) + b) % b)),
        Arguments.of(BinaryOperator.POW, in1, in2, mapArray(in1, in2,
            (a, b) -> (float) Math.pow(a, b))));
  }

  @ParameterizedTest
  @MethodSource("createBinaryFloatOperationParams")
  void testBinaryFloatOperationTransform(BinaryOperator op, Float[] in1, Float[] in2,
                                          List<Float> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedFloats(
             expectedValues.toArray(new Float[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryDoublePromotedOperationParams() {
    Float[] in1 = new Float[] { -5f, 4.5f, null, 2.7f };
    Float[] in2 = new Float[] { 123f, -456f, null, 0f };
    return Stream.of(
        Arguments.of(BinaryOperator.TRUE_DIV, in1, in2, mapArray(in1, in2,
            (a, b) -> (double) a / b)),
        Arguments.of(BinaryOperator.FLOOR_DIV, in1, in2, mapArray(in1, in2,
            (a, b) -> Math.floor(a / b))));
  }

  @ParameterizedTest
  @MethodSource("createBinaryDoublePromotedOperationParams")
  void testBinaryDoublePromotedOperationTransform(BinaryOperator op, Float[] in1, Float[] in2,
                                                   List<Double> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(
             expectedValues.toArray(new Double[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryComparisonOperationParams() {
    Integer[] in1 = new Integer[] { -5, 4, null, 2, -3 };
    Integer[] in2 = new Integer[] { 123, -456, null, 0, -3 };
    return Stream.of(
        // nulls compare as equal by default
        Arguments.of(BinaryOperator.NULL_EQUAL, in1, in2, Arrays.asList(false, false, true, false, true)),
        Arguments.of(BinaryOperator.NOT_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> !a.equals(b))),
        Arguments.of(BinaryOperator.LESS, in1, in2, mapArray(in1, in2, (a, b) -> a < b)),
        Arguments.of(BinaryOperator.GREATER, in1, in2, mapArray(in1, in2, (a, b) -> a > b)),
        Arguments.of(BinaryOperator.LESS_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> a <= b)),
        Arguments.of(BinaryOperator.GREATER_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> a >= b)));
  }

  @ParameterizedTest
  @MethodSource("createBinaryComparisonOperationParams")
  void testBinaryComparisonOperationTransform(BinaryOperator op, Integer[] in1, Integer[] in2,
                                               List<Boolean> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createStringLiteralComparisonParams() {
    String[] in1 = new String[] {"a", "bb", null, "ccc", "dddd"};
    String in2 = "ccc";
    return Stream.of(
        // nulls compare as equal by default
        Arguments.of(BinaryOperator.NULL_EQUAL, in1, in2, Arrays.asList(false, false, false, true, false)),
        Arguments.of(BinaryOperator.NOT_EQUAL, in1, in2, mapArray(in1, (a) -> !a.equals(in2))),
        Arguments.of(BinaryOperator.LESS, in1, in2, mapArray(in1, (a) -> a.compareTo(in2) < 0)),
        Arguments.of(BinaryOperator.GREATER, in1, in2, mapArray(in1, (a) -> a.compareTo(in2) > 0)),
        Arguments.of(BinaryOperator.LESS_EQUAL, in1, in2, mapArray(in1, (a) -> a.compareTo(in2) <= 0)),
        Arguments.of(BinaryOperator.GREATER_EQUAL, in1, in2, mapArray(in1, (a) -> a.compareTo(in2) >= 0)),
        // null literal
        Arguments.of(BinaryOperator.NULL_EQUAL, in1, null, Arrays.asList(false, false, true, false, false)),
        Arguments.of(BinaryOperator.NOT_EQUAL, in1, null, Arrays.asList(null, null, null, null, null)),
        Arguments.of(BinaryOperator.LESS, in1, null, Arrays.asList(null, null, null, null, null)));
  }

  @ParameterizedTest
  @MethodSource("createStringLiteralComparisonParams")
  void testStringLiteralComparison(BinaryOperator op, String[] in1, String in2,
                                               List<Boolean> expectedValues) {
    Literal lit = Literal.ofString(in2);
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        lit);
    try (Table t = new Table.TestBuilder().column(in1).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryComparisonOperationStringParams() {
    String[] in1 = new String[] {"a", "bb", null, "ccc", "dddd"};
    String[] in2 = new String[] {"aa", "b", null, "ccc", "ddd"};
    return Stream.of(
        // nulls compare as equal by default
        Arguments.of(BinaryOperator.NULL_EQUAL, in1, in2, Arrays.asList(false, false, true, true, false)),
        Arguments.of(BinaryOperator.NOT_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> !a.equals(b))),
        Arguments.of(BinaryOperator.LESS, in1, in2, mapArray(in1, in2, (a, b) -> a.compareTo(b) < 0)),
        Arguments.of(BinaryOperator.GREATER, in1, in2, mapArray(in1, in2, (a, b) -> a.compareTo(b) > 0)),
        Arguments.of(BinaryOperator.LESS_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> a.compareTo(b) <= 0)),
        Arguments.of(BinaryOperator.GREATER_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> a.compareTo(b) >= 0)));
  }

  @ParameterizedTest
  @MethodSource("createBinaryComparisonOperationStringParams")
  void testBinaryComparisonOperationStringTransform(BinaryOperator op, String[] in1, String[] in2,
                                               List<Boolean> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryBitwiseOperationParams() {
    Integer[] in1 = new Integer[] { -5, 4, null, 2, -3 };
    Integer[] in2 = new Integer[] { 123, -456, null, 0, -3 };
    return Stream.of(
        Arguments.of(BinaryOperator.BITWISE_AND, in1, in2, mapArray(in1, in2, (a, b) -> a & b)),
        Arguments.of(BinaryOperator.BITWISE_OR, in1, in2, mapArray(in1, in2, (a, b) -> a | b)),
        Arguments.of(BinaryOperator.BITWISE_XOR, in1, in2, mapArray(in1, in2, (a, b) -> a ^ b)));
  }

  @ParameterizedTest
  @MethodSource("createBinaryBitwiseOperationParams")
  void testBinaryBitwiseOperationTransform(BinaryOperator op, Integer[] in1, Integer[] in2,
                                            List<Integer> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedInts(
             expectedValues.toArray(new Integer[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryBooleanOperationParams() {
    Boolean[] in1 = new Boolean[] { false, true, false, null, true, false };
    Boolean[] in2 = new Boolean[] { true, null, null, null, true, false };
    return Stream.of(
        Arguments.of(BinaryOperator.LOGICAL_AND, in1, in2, mapArray(in1, in2, (a, b) -> a && b)),
        Arguments.of(BinaryOperator.LOGICAL_OR, in1, in2, mapArray(in1, in2, (a, b) -> a || b)),
        Arguments.of(BinaryOperator.NULL_LOGICAL_AND, in1, in2, Arrays.asList(false, null, false, null, true, false)),
        Arguments.of(BinaryOperator.NULL_LOGICAL_OR, in1, in2, Arrays.asList(true, true, null, null, true, false)));
  }

  @ParameterizedTest
  @MethodSource("createBinaryBooleanOperationParams")
  void testBinaryBooleanOperationTransform(BinaryOperator op, Boolean[] in1, Boolean[] in2,
                                            List<Boolean> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testMismatchedBinaryOperationTypes() {
    // verify expression fails to transform if operands are not the same type
    BinaryOperation expr = new BinaryOperation(BinaryOperator.ADD,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(1, 2, 3).column(1L, 2L, 3L).build();
         CompiledExpression compiledExpr = expr.compile()) {
      Assertions.assertThrows(CudfException.class, () -> compiledExpr.computeColumn(t).close());
    }
  }
}
