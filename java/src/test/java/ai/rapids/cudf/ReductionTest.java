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

  private static Scalar buildExpectedScalar(Aggregation op, DType baseType, Object expectedObject) {
    if (expectedObject == null) {
      return Scalar.fromNull(baseType);
    }
    if (FLOAT_REDUCTIONS.contains(op.kind)) {
      if (baseType == DType.FLOAT32) {
        return Scalar.fromFloat((Float) expectedObject);
      }
      return Scalar.fromDouble((Double) expectedObject);
    }
    if (BOOL_REDUCTIONS.contains(op.kind)) {
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
        Arguments.of(Aggregation.sum(), new Boolean[0], null, 0.),
        Arguments.of(Aggregation.sum(), new Boolean[]{null, null, null}, null, 0.),
        Arguments.of(Aggregation.sum(), vals, true, 0.),
        Arguments.of(Aggregation.min(), vals, false, 0.),
        Arguments.of(Aggregation.max(), vals, true, 0.),
        Arguments.of(Aggregation.product(), vals, false, 0.),
        Arguments.of(Aggregation.sumOfSquares(), vals, true, 0.),
        Arguments.of(Aggregation.mean(), vals, 0.6, DELTAD),
        Arguments.of(Aggregation.standardDeviation(), vals, 0.5477225575051662, DELTAD),
        Arguments.of(Aggregation.variance(), vals, 0.3, DELTAD),
        Arguments.of(Aggregation.any(), vals, true, 0.),
        Arguments.of(Aggregation.all(), vals, false, 0.)
    );
  }

  private static Stream<Arguments> createByteParams() {
    Byte[] vals = new Byte[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(Aggregation.sum(), new Byte[0], null, 0.),
        Arguments.of(Aggregation.sum(), new Byte[]{null, null, null}, null, 0.),
        Arguments.of(Aggregation.sum(), vals, (byte) 83, 0.),
        Arguments.of(Aggregation.min(), vals, (byte) -1, 0.),
        Arguments.of(Aggregation.max(), vals, (byte) 123, 0.),
        Arguments.of(Aggregation.product(), vals, (byte) 160, 0.),
        Arguments.of(Aggregation.sumOfSquares(), vals, (byte) 47, 0.),
        Arguments.of(Aggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(Aggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(Aggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(Aggregation.any(), vals, true, 0.),
        Arguments.of(Aggregation.all(), vals, true, 0.)
    );
  }

  private static Stream<Arguments> createShortParams() {
    Short[] vals = new Short[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(Aggregation.sum(), new Short[0], null, 0.),
        Arguments.of(Aggregation.sum(), new Short[]{null, null, null}, null, 0.),
        Arguments.of(Aggregation.sum(), vals, (short) 339, 0.),
        Arguments.of(Aggregation.min(), vals, (short) -1, 0.),
        Arguments.of(Aggregation.max(), vals, (short) 123, 0.),
        Arguments.of(Aggregation.product(), vals, (short) -22624, 0.),
        Arguments.of(Aggregation.sumOfSquares(), vals, (short) 31279, 0.),
        Arguments.of(Aggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(Aggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(Aggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(Aggregation.any(), vals, true, 0.),
        Arguments.of(Aggregation.all(), vals, true, 0.)
    );
  }

  private static Stream<Arguments> createIntParams() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(Aggregation.sum(), new Integer[0], null, 0.),
        Arguments.of(Aggregation.sum(), new Integer[]{null, null, null}, null, 0.),
        Arguments.of(Aggregation.sum(), vals, 339, 0.),
        Arguments.of(Aggregation.min(), vals, -1, 0.),
        Arguments.of(Aggregation.max(), vals, 123, 0.),
        Arguments.of(Aggregation.product(), vals, -258300000, 0.),
        Arguments.of(Aggregation.sumOfSquares(), vals, 31279, 0.),
        Arguments.of(Aggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(Aggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(Aggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(Aggregation.any(), vals, true, 0.),
        Arguments.of(Aggregation.all(), vals, true, 0.)
    );
  }

  private static Stream<Arguments> createLongParams() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    return Stream.of(
        Arguments.of(Aggregation.sum(), new Long[0], null, 0.),
        Arguments.of(Aggregation.sum(), new Long[]{null, null, null}, null, 0.),
        Arguments.of(Aggregation.sum(), vals, 339L, 0.),
        Arguments.of(Aggregation.min(), vals, -1L, 0.),
        Arguments.of(Aggregation.max(), vals, 123L, 0.),
        Arguments.of(Aggregation.product(), vals, -258300000L, 0.),
        Arguments.of(Aggregation.sumOfSquares(), vals, 31279L, 0.),
        Arguments.of(Aggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(Aggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(Aggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(Aggregation.any(), vals, true, 0.),
        Arguments.of(Aggregation.all(), vals, true, 0.),
        Arguments.of(Aggregation.quantile(0.5), vals, 55.0, DELTAD),
        Arguments.of(Aggregation.quantile(0.9), vals, 111.5, DELTAD)
    );
  }

  private static Stream<Arguments> createFloatParams() {
    Float[] vals = new Float[]{-1f, 7f, 123f, null, 50f, 60f, 100f};
    return Stream.of(
        Arguments.of(Aggregation.sum(), new Float[0], null, 0f),
        Arguments.of(Aggregation.sum(), new Float[]{null, null, null}, null, 0f),
        Arguments.of(Aggregation.sum(), vals, 339f, 0f),
        Arguments.of(Aggregation.min(), vals, -1f, 0f),
        Arguments.of(Aggregation.max(), vals, 123f, 0f),
        Arguments.of(Aggregation.product(), vals, -258300000f, 0f),
        Arguments.of(Aggregation.sumOfSquares(), vals, 31279f, 0f),
        Arguments.of(Aggregation.mean(), vals, 56.5f, DELTAF),
        Arguments.of(Aggregation.standardDeviation(), vals, 49.24530434467839f, DELTAF),
        Arguments.of(Aggregation.variance(), vals, 2425.1f, DELTAF),
        Arguments.of(Aggregation.any(), vals, true, 0f),
        Arguments.of(Aggregation.all(), vals, true, 0f)
    );
  }

  private static Stream<Arguments> createDoubleParams() {
    Double[] vals = new Double[]{-1., 7., 123., null, 50., 60., 100.};
    return Stream.of(
        Arguments.of(Aggregation.sum(), new Double[0], null, 0.),
        Arguments.of(Aggregation.sum(), new Double[]{null, null, null}, null, 0.),
        Arguments.of(Aggregation.sum(), vals, 339., 0.),
        Arguments.of(Aggregation.min(), vals, -1., 0.),
        Arguments.of(Aggregation.max(), vals, 123., 0.),
        Arguments.of(Aggregation.product(), vals, -258300000., 0.),
        Arguments.of(Aggregation.sumOfSquares(), vals, 31279., 0.),
        Arguments.of(Aggregation.mean(), vals, 56.5, DELTAD),
        Arguments.of(Aggregation.standardDeviation(), vals, 49.24530434467839, DELTAD),
        Arguments.of(Aggregation.variance(), vals, 2425.1, DELTAD),
        Arguments.of(Aggregation.any(), vals, true, 0.),
        Arguments.of(Aggregation.all(), vals, true, 0.),
        Arguments.of(Aggregation.quantile(0.5), vals, 55.0, DELTAD),
        Arguments.of(Aggregation.quantile(0.9), vals, 111.5, DELTAD)
    );
  }

  private static Stream<Arguments> createTimestampDaysParams() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(Aggregation.max(), new Integer[0], null),
        Arguments.of(Aggregation.max(), new Integer[]{null, null, null}, null),
        Arguments.of(Aggregation.max(), vals, 123),
        Arguments.of(Aggregation.min(), vals, -1)
    );
  }

  private static Stream<Arguments> createTimestampResolutionParams() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    return Stream.of(
        Arguments.of(Aggregation.max(), new Long[0], null),
        Arguments.of(Aggregation.max(), new Long[]{null, null, null}, null),
        Arguments.of(Aggregation.min(), vals, -1L),
        Arguments.of(Aggregation.max(), vals, 123L)
    );
  }

  private static void assertEqualsDelta(Aggregation op, Scalar expected, Scalar result,
                                        Double percentage) {
    if (FLOAT_REDUCTIONS.contains(op.kind)) {
      assertEqualsWithinPercentage(expected.getDouble(), result.getDouble(), percentage);
    } else {
      assertEquals(expected, result);
    }
  }

  private static void assertEqualsDelta(Aggregation op, Scalar expected, Scalar result,
                                        Float percentage) {
    if (FLOAT_REDUCTIONS.contains(op.kind)) {
      assertEqualsWithinPercentage(expected.getFloat(), result.getFloat(), percentage);
    } else {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createBooleanParams")
  void testBoolean(Aggregation op, Boolean[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.BOOL8, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedBooleans(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createByteParams")
  void testByte(Aggregation op, Byte[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT8, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedBytes(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createShortParams")
  void testShort(Aggregation op, Short[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT16, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedShorts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntParams")
  void testInt(Aggregation op, Integer[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT32, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedInts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongParams")
  void testLong(Aggregation op, Long[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT64, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatParams")
  void testFloat(Aggregation op, Float[] values, Object expectedObject, Float delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.FLOAT32, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedFloats(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createDoubleParams")
  void testDouble(Aggregation op, Double[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.FLOAT64, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedDoubles(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampDaysParams")
  void testTimestampDays(Aggregation op, Integer[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_DAYS, expectedObject);
         ColumnVector v = ColumnVector.timestampDaysFromBoxedInts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampSeconds(Aggregation op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_SECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampMilliseconds(Aggregation op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_MILLISECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampMilliSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampMicroseconds(Aggregation op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_MICROSECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampMicroSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampNanoseconds(Aggregation op, Long[] values, Object expectedObject) {
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
