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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.EnumSet;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ReductionTest extends CudfTestBase {
  public static final double DELTAD = 0.00001;
  public static final float DELTAF = 0.001f;

  // reduction operations that produce a floating point value
  private static final EnumSet<Aggregation.Kind> FLOAT_REDUCTIONS = EnumSet.of(
      Aggregation.Kind.MEAN,
      Aggregation.Kind.STD,
      Aggregation.Kind.VARIANCE,
      Aggregation.Kind.QUANTILE);

  // reduction operations that produce a floating point value
  private static final EnumSet<Aggregation.Kind> BOOL_REDUCTIONS = EnumSet.of(
      Aggregation.Kind.ANY,
      Aggregation.Kind.ALL);

  private static Scalar buildExpectedScalar(ReductionAggregation op, DType baseType, Object expectedObject) {
    if (expectedObject == null) {
      return Scalar.fromNull(baseType);
    }
    if (FLOAT_REDUCTIONS.contains(op.getWrapped().kind)) {
      if (baseType.equals(DType.FLOAT32)) {
        return Scalar.fromFloat((Float) expectedObject);
      }
      return Scalar.fromDouble((Double) expectedObject);
    }
    if (BOOL_REDUCTIONS.contains(op.getWrapped().kind)) {
      return Scalar.fromBool((Boolean) expectedObject);
    }
    switch (baseType.typeId) {
    case BOOL8:
      return Scalar.fromBool((Boolean) expectedObject);
    case INT8:
      return Scalar.fromByte((Byte) expectedObject);
    case INT16:
      return Scalar.fromShort((Short) expectedObject);
    case INT32:
      return Scalar.fromInt((Integer) expectedObject);
    case INT64:
      return Scalar.fromLong((Long) expectedObject);
    case FLOAT32:
      return Scalar.fromFloat((Float) expectedObject);
    case FLOAT64:
      return Scalar.fromDouble((Double) expectedObject);
    case TIMESTAMP_DAYS:
      return Scalar.timestampDaysFromInt((Integer) expectedObject);
    case TIMESTAMP_SECONDS:
    case TIMESTAMP_MILLISECONDS:
    case TIMESTAMP_MICROSECONDS:
    case TIMESTAMP_NANOSECONDS:
      return Scalar.timestampFromLong(baseType, (Long) expectedObject);
    case STRING:
      return Scalar.fromString((String) expectedObject);
    default:
      throw new IllegalArgumentException("Unexpected type: " + baseType);
    }
  }

  private static Stream<Arguments> createBooleanParams() {
    Boolean[] vals = new Boolean[]{true, true, null, false, true, false, null};
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Boolean[0], null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Boolean[]{null, null, null}, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, true, 0.),
        Arguments.of(ReductionAggregation.min(), vals, false, 0.),
        Arguments.of(ReductionAggregation.max(), vals, true, 0.),
        Arguments.of(ReductionAggregation.product(), vals, false, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, true, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, 0.6, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, 0.5477225575051662, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, 0.3, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, false, 0.)
    );
  }

  private static Stream<Arguments> createByteParams() {
    Byte[] vals = new Byte[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Byte[0], null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Byte[]{null, null, null}, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, (byte) 83, 0.),
        Arguments.of(ReductionAggregation.min(), vals, (byte) -1, 0.),
        Arguments.of(ReductionAggregation.max(), vals, (byte) 123, 0.),
        Arguments.of(ReductionAggregation.product(), vals, (byte) 160, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, (byte) 47, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, true, 0.)
    );
  }

  private static Stream<Arguments> createShortParams() {
    Short[] vals = new Short[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Short[0], null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Short[]{null, null, null}, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, (short) 339, 0.),
        Arguments.of(ReductionAggregation.min(), vals, (short) -1, 0.),
        Arguments.of(ReductionAggregation.max(), vals, (short) 123, 0.),
        Arguments.of(ReductionAggregation.product(), vals, (short) -22624, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, (short) 31279, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, true, 0.)
    );
  }

  private static Stream<Arguments> createIntParams() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Integer[0], null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Integer[]{null, null, null}, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, 339, 0.),
        Arguments.of(ReductionAggregation.min(), vals, -1, 0.),
        Arguments.of(ReductionAggregation.max(), vals, 123, 0.),
        Arguments.of(ReductionAggregation.product(), vals, -258300000, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, 31279, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, true, 0.)
    );
  }

  private static Stream<Arguments> createLongParams() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Long[0], null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Long[]{null, null, null}, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, 339L, 0.),
        Arguments.of(ReductionAggregation.min(), vals, -1L, 0.),
        Arguments.of(ReductionAggregation.max(), vals, 123L, 0.),
        Arguments.of(ReductionAggregation.product(), vals, -258300000L, 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, 31279L, 0.),
        Arguments.of(ReductionAggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, true, 0.),
        Arguments.of(ReductionAggregation.quantile(0.5), vals, 55.0, DELTAD),
        Arguments.of(ReductionAggregation.quantile(0.9), vals, 111.5, DELTAD)
    );
  }

  private static Stream<Arguments> createFloatParams() {
    Float[] vals = new Float[]{-1f, 7f, 123f, null, 50f, 60f, 100f};
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Float[0], null, 0f),
        Arguments.of(ReductionAggregation.sum(), new Float[]{null, null, null}, null, 0f),
        Arguments.of(ReductionAggregation.sum(), vals, 339f, 0f),
        Arguments.of(ReductionAggregation.min(), vals, -1f, 0f),
        Arguments.of(ReductionAggregation.max(), vals, 123f, 0f),
        Arguments.of(ReductionAggregation.product(), vals, -258300000f, 0f),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, 31279f, 0f),
        Arguments.of(ReductionAggregation.mean(), vals, 56.5f, DELTAF),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, 49.24530434467839f, DELTAF),
        Arguments.of(ReductionAggregation.variance(), vals, 2425.1f, DELTAF),
        Arguments.of(ReductionAggregation.any(), vals, true, 0f),
        Arguments.of(ReductionAggregation.all(), vals, true, 0f)
    );
  }

  private static Stream<Arguments> createDoubleParams() {
    Double[] vals = new Double[]{-1., 7., 123., null, 50., 60., 100.};
    return Stream.of(
        Arguments.of(ReductionAggregation.sum(), new Double[0], null, 0.),
        Arguments.of(ReductionAggregation.sum(), new Double[]{null, null, null}, null, 0.),
        Arguments.of(ReductionAggregation.sum(), vals, 339., 0.),
        Arguments.of(ReductionAggregation.min(), vals, -1., 0.),
        Arguments.of(ReductionAggregation.max(), vals, 123., 0.),
        Arguments.of(ReductionAggregation.product(), vals, -258300000., 0.),
        Arguments.of(ReductionAggregation.sumOfSquares(), vals, 31279., 0.),
        Arguments.of(ReductionAggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(ReductionAggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(ReductionAggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(ReductionAggregation.any(), vals, true, 0.),
        Arguments.of(ReductionAggregation.all(), vals, true, 0.),
        Arguments.of(ReductionAggregation.quantile(0.5), vals, 55.0, DELTAD),
        Arguments.of(ReductionAggregation.quantile(0.9), vals, 111.5, DELTAD)
    );
  }

  private static Stream<Arguments> createTimestampDaysParams() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionAggregation.max(), new Integer[0], null),
        Arguments.of(ReductionAggregation.max(), new Integer[]{null, null, null}, null),
        Arguments.of(ReductionAggregation.max(), vals, 123),
        Arguments.of(ReductionAggregation.min(), vals, -1)
    );
  }

  private static Stream<Arguments> createTimestampResolutionParams() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    return Stream.of(
        Arguments.of(ReductionAggregation.max(), new Long[0], null),
        Arguments.of(ReductionAggregation.max(), new Long[]{null, null, null}, null),
        Arguments.of(ReductionAggregation.min(), vals, -1L),
        Arguments.of(ReductionAggregation.max(), vals, 123L)
    );
  }

  private static void assertEqualsDelta(ReductionAggregation op, Scalar expected, Scalar result,
                                        Double percentage) {
    if (FLOAT_REDUCTIONS.contains(op.getWrapped().kind)) {
      assertEqualsWithinPercentage(expected.getDouble(), result.getDouble(), percentage);
    } else {
      assertEquals(expected, result);
    }
  }

  private static void assertEqualsDelta(ReductionAggregation op, Scalar expected, Scalar result,
                                        Float percentage) {
    if (FLOAT_REDUCTIONS.contains(op.getWrapped().kind)) {
      assertEqualsWithinPercentage(expected.getFloat(), result.getFloat(), percentage);
    } else {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createBooleanParams")
  void testBoolean(ReductionAggregation op, Boolean[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.BOOL8, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedBooleans(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createByteParams")
  void testByte(ReductionAggregation op, Byte[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT8, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedBytes(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createShortParams")
  void testShort(ReductionAggregation op, Short[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT16, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedShorts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntParams")
  void testInt(ReductionAggregation op, Integer[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT32, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedInts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongParams")
  void testLong(ReductionAggregation op, Long[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT64, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatParams")
  void testFloat(ReductionAggregation op, Float[] values, Object expectedObject, Float delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.FLOAT32, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedFloats(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createDoubleParams")
  void testDouble(ReductionAggregation op, Double[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.FLOAT64, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedDoubles(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampDaysParams")
  void testTimestampDays(ReductionAggregation op, Integer[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_DAYS, expectedObject);
         ColumnVector v = ColumnVector.timestampDaysFromBoxedInts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampSeconds(ReductionAggregation op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_SECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampMilliseconds(ReductionAggregation op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_MILLISECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampMilliSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampMicroseconds(ReductionAggregation op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_MICROSECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampMicroSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampNanoseconds(ReductionAggregation op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_NANOSECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampNanoSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @Test
  void testWithSetOutputType() {
    try (Scalar expected = Scalar.fromLong(1 * 2 * 3 * 4L);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.product(DType.INT64)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromLong(1 + 2 + 3 + 4L);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.sum(DType.INT64)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromLong((1*1L) + (2*2L) + (3*3L) + (4*4L));
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.sumOfSquares(DType.INT64)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromFloat((1 + 2 + 3 + 4f)/4);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.mean(DType.FLOAT32)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromFloat(1.666667f);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.variance(DType.FLOAT32)) {
      assertEquals(expected, result);
    }

    try (Scalar expected = Scalar.fromFloat(1.2909945f);
         ColumnVector cv = ColumnVector.fromBytes(new byte[]{1, 2, 3, 4});
         Scalar result = cv.standardDeviation(DType.FLOAT32)) {
      assertEquals(expected, result);
    }
  }
}
