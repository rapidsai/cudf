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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ReductionTest extends CudfTestBase {

  public static final double DELTAD = 0.00001;
  public static final float DELTAF = 0.00001f;

  private static Stream<Arguments> createBooleanParams() {
    Boolean[] vals = new Boolean[]{true, true, null, false, true, false, null};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Boolean[0], Scalar.fromNull(DType.BOOL8), 0.),
        Arguments.of(ReductionOp.SUM, new Boolean[]{null, null, null},
            Scalar.fromNull(DType.BOOL8), 0.),
        Arguments.of(ReductionOp.SUM, vals, Scalar.fromBool(true), 0.),
        Arguments.of(ReductionOp.MIN, vals, Scalar.fromBool(false), 0.),
        Arguments.of(ReductionOp.MAX, vals, Scalar.fromBool(true), 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, Scalar.fromBool(false), 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, Scalar.fromBool(true), 0.),
        Arguments.of(ReductionOp.STD, vals, Scalar.fromDouble(0.5477225575051662), DELTAD)
    );
  }

  private static Stream<Arguments> createByteParams() {
    Byte[] vals = new Byte[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Byte[0], Scalar.fromNull(DType.INT8), 0.),
        Arguments.of(ReductionOp.SUM, new Byte[]{null, null, null}, Scalar.fromNull(DType.INT8), 0.),
        Arguments.of(ReductionOp.SUM, vals, Scalar.fromByte((byte) 83), 0.),
        Arguments.of(ReductionOp.MIN, vals, Scalar.fromByte((byte) -1), 0.),
        Arguments.of(ReductionOp.MAX, vals, Scalar.fromByte((byte) 123), 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, Scalar.fromByte((byte) 160), 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, Scalar.fromByte((byte) 47), 0.),
        Arguments.of(ReductionOp.STD, vals, Scalar.fromDouble(49.24530434467839), DELTAD)
    );
  }

  private static Stream<Arguments> createShortParams() {
    Short[] vals = new Short[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Short[0], Scalar.fromNull(DType.INT16), 0.),
        Arguments.of(ReductionOp.SUM, new Short[]{null, null, null}, Scalar.fromNull(DType.INT16), 0.),
        Arguments.of(ReductionOp.SUM, vals, Scalar.fromShort((short) 339), 0.),
        Arguments.of(ReductionOp.MIN, vals, Scalar.fromShort((short) -1), 0.),
        Arguments.of(ReductionOp.MAX, vals, Scalar.fromShort((short) 123), 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, Scalar.fromShort((short) -22624), 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, Scalar.fromShort((short) 31279), 0.),
        Arguments.of(ReductionOp.STD, vals, Scalar.fromDouble(49.24530434467839), DELTAD)
    );
  }

  private static Stream<Arguments> createIntParams() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Integer[0], Scalar.fromNull(DType.INT32), 0.),
        Arguments.of(ReductionOp.SUM, new Integer[]{null, null, null},
            Scalar.fromNull(DType.INT32), 0.),
        Arguments.of(ReductionOp.SUM, vals, Scalar.fromInt(339), 0.),
        Arguments.of(ReductionOp.MIN, vals, Scalar.fromInt(-1), 0.),
        Arguments.of(ReductionOp.MAX, vals, Scalar.fromInt(123), 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, Scalar.fromInt(-258300000), 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, Scalar.fromInt(31279), 0.),
        Arguments.of(ReductionOp.STD, vals, Scalar.fromDouble(49.24530434467839), DELTAD)
    );
  }

  private static Stream<Arguments> createLongParams() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Long[0], Scalar.fromNull(DType.INT64), 0.),
        Arguments.of(ReductionOp.SUM, new Long[]{null, null, null}, Scalar.fromNull(DType.INT64), 0.),
        Arguments.of(ReductionOp.SUM, vals, Scalar.fromLong(339), 0.),
        Arguments.of(ReductionOp.MIN, vals, Scalar.fromLong(-1), 0.),
        Arguments.of(ReductionOp.MAX, vals, Scalar.fromLong(123), 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, Scalar.fromLong(-258300000), 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, Scalar.fromLong(31279), 0.),
        Arguments.of(ReductionOp.STD, vals, Scalar.fromDouble(49.24530434467839), DELTAD)
    );
  }

  private static Stream<Arguments> createFloatParams() {
    Float[] vals = new Float[]{-1f, 7f, 123f, null, 50f, 60f, 100f};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Float[0], Scalar.fromNull(DType.FLOAT32), 0f),
        Arguments.of(ReductionOp.SUM, new Float[]{null, null, null},
            Scalar.fromNull(DType.FLOAT32), 0f),
        Arguments.of(ReductionOp.SUM, vals, Scalar.fromFloat(339f), 0f),
        Arguments.of(ReductionOp.MIN, vals, Scalar.fromFloat(-1f), 0f),
        Arguments.of(ReductionOp.MAX, vals, Scalar.fromFloat(123f), 0f),
        Arguments.of(ReductionOp.PRODUCT, vals, Scalar.fromFloat(-258300000f), 0f),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, Scalar.fromFloat(31279f), 0f),
        Arguments.of(ReductionOp.STD, vals, Scalar.fromFloat(49.24530434467839f), DELTAF)
    );
  }

  private static Stream<Arguments> createDoubleParams() {
    Double[] vals = new Double[]{-1., 7., 123., null, 50., 60., 100.};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Double[0], Scalar.fromNull(DType.FLOAT64), 0.),
        Arguments.of(ReductionOp.SUM, new Double[]{null, null, null},
            Scalar.fromNull(DType.FLOAT64), 0.),
        Arguments.of(ReductionOp.SUM, vals, Scalar.fromDouble(339.), 0.),
        Arguments.of(ReductionOp.MIN, vals, Scalar.fromDouble(-1.), 0.),
        Arguments.of(ReductionOp.MAX, vals, Scalar.fromDouble(123.), 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, Scalar.fromDouble(-258300000.), 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, Scalar.fromDouble(31279.), 0.),
        Arguments.of(ReductionOp.STD, vals, Scalar.fromDouble(49.24530434467839), DELTAD)
    );
  }

  private static Stream<Arguments> createDate32Params() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionOp.MAX, new Integer[0], Scalar.fromNull(DType.DATE32)),
        Arguments.of(ReductionOp.MAX, new Integer[]{null, null, null},
            Scalar.fromNull(DType.DATE32)),
        Arguments.of(ReductionOp.MAX, vals, Scalar.dateFromInt(123)),
        Arguments.of(ReductionOp.MIN, vals, Scalar.dateFromInt(-1))
    );
  }

  private static Stream<Arguments> createDate64Params() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    return Stream.of(
        Arguments.of(ReductionOp.MAX, new Long[0], Scalar.fromNull(DType.DATE64)),
        Arguments.of(ReductionOp.MAX, new Long[]{null, null, null}, Scalar.fromNull(DType.DATE64)),
        Arguments.of(ReductionOp.MIN, vals, Scalar.dateFromLong(-1)),
        Arguments.of(ReductionOp.MAX, vals, Scalar.dateFromLong(123))
    );
  }

  private static Stream<Arguments> createTimestampParams() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    return Stream.of(
        Arguments.of(ReductionOp.MAX, new Long[0], TimeUnit.NONE,
            Scalar.timestampFromNull(TimeUnit.MILLISECONDS)),
        Arguments.of(ReductionOp.MAX, new Long[]{null, null, null}, TimeUnit.MICROSECONDS,
            Scalar.timestampFromNull(TimeUnit.MICROSECONDS)),
        Arguments.of(ReductionOp.MIN, vals, TimeUnit.SECONDS,
            Scalar.timestampFromLong(-1, TimeUnit.SECONDS)),
        Arguments.of(ReductionOp.MAX, vals, TimeUnit.NANOSECONDS,
            Scalar.timestampFromLong(123, TimeUnit.NANOSECONDS))
    );
  }

  @ParameterizedTest
  @MethodSource("createBooleanParams")
  void testBoolean(ReductionOp op, Boolean[] values, Scalar expected, Double delta) {
    try (ColumnVector v = ColumnVector.fromBoxedBooleans(values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected.intTypeStorage, result.intTypeStorage);
      assertEquals(expected.floatTypeStorage, result.floatTypeStorage);
      assertEquals(expected.doubleTypeStorage, result.doubleTypeStorage, delta);
      assertEquals(expected.stringTypeStorage, result.stringTypeStorage);
      assertEquals(expected.type, result.type);
      assertEquals(expected.isValid, result.isValid);
      assertEquals(expected.timeUnit, result.timeUnit);
    }
  }

  @ParameterizedTest
  @MethodSource("createByteParams")
  void testByte(ReductionOp op, Byte[] values, Scalar expected, Double delta) {
    try (ColumnVector v = ColumnVector.fromBoxedBytes(values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected.intTypeStorage, result.intTypeStorage);
      assertEquals(expected.floatTypeStorage, result.floatTypeStorage);
      assertEquals(expected.doubleTypeStorage, result.doubleTypeStorage, delta);
      assertEquals(expected.stringTypeStorage, result.stringTypeStorage);
      assertEquals(expected.type, result.type);
      assertEquals(expected.isValid, result.isValid);
      assertEquals(expected.timeUnit, result.timeUnit);
    }
  }

  @ParameterizedTest
  @MethodSource("createShortParams")
  void testShort(ReductionOp op, Short[] values, Scalar expected, Double delta) {
    try (ColumnVector v = ColumnVector.fromBoxedShorts(values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected.intTypeStorage, result.intTypeStorage);
      assertEquals(expected.floatTypeStorage, result.floatTypeStorage);
      assertEquals(expected.doubleTypeStorage, result.doubleTypeStorage, delta);
      assertEquals(expected.stringTypeStorage, result.stringTypeStorage);
      assertEquals(expected.type, result.type);
      assertEquals(expected.isValid, result.isValid);
      assertEquals(expected.timeUnit, result.timeUnit);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntParams")
  void testInt(ReductionOp op, Integer[] values, Scalar expected, Double delta) {
    try (ColumnVector v = ColumnVector.fromBoxedInts(values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected.intTypeStorage, result.intTypeStorage);
      assertEquals(expected.floatTypeStorage, result.floatTypeStorage);
      assertEquals(expected.doubleTypeStorage, result.doubleTypeStorage, delta);
      assertEquals(expected.stringTypeStorage, result.stringTypeStorage);
      assertEquals(expected.type, result.type);
      assertEquals(expected.isValid, result.isValid);
      assertEquals(expected.timeUnit, result.timeUnit);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongParams")
  void testLong(ReductionOp op, Long[] values, Scalar expected, Double delta) {
    try (ColumnVector v = ColumnVector.fromBoxedLongs(values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected.intTypeStorage, result.intTypeStorage);
      assertEquals(expected.floatTypeStorage, result.floatTypeStorage);
      assertEquals(expected.doubleTypeStorage, result.doubleTypeStorage, delta);
      assertEquals(expected.stringTypeStorage, result.stringTypeStorage);
      assertEquals(expected.type, result.type);
      assertEquals(expected.isValid, result.isValid);
      assertEquals(expected.timeUnit, result.timeUnit);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatParams")
  void testFloat(ReductionOp op, Float[] values, Scalar expected, Float delta) {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected.intTypeStorage, result.intTypeStorage);
      assertEquals(expected.floatTypeStorage, result.floatTypeStorage, delta);
      assertEquals(expected.doubleTypeStorage, result.doubleTypeStorage);
      assertEquals(expected.stringTypeStorage, result.stringTypeStorage);
      assertEquals(expected.type, result.type);
      assertEquals(expected.isValid, result.isValid);
      assertEquals(expected.timeUnit, result.timeUnit);
    }
  }

  @ParameterizedTest
  @MethodSource("createDoubleParams")
  void testByte(ReductionOp op, Double[] values, Scalar expected, Double delta) {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected.intTypeStorage, result.intTypeStorage);
      assertEquals(expected.floatTypeStorage, result.floatTypeStorage);
      assertEquals(expected.doubleTypeStorage, result.doubleTypeStorage, delta);
      assertEquals(expected.stringTypeStorage, result.stringTypeStorage);
      assertEquals(expected.type, result.type);
      assertEquals(expected.isValid, result.isValid);
      assertEquals(expected.timeUnit, result.timeUnit);
    }
  }

  @ParameterizedTest
  @MethodSource("createDate32Params")
  void testDate32(ReductionOp op, Integer[] values, Scalar expected) {
    try (ColumnVector v = ColumnVector.datesFromBoxedInts(values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createDate64Params")
  void testDate64(ReductionOp op, Long[] values, Scalar expected) {
    try (ColumnVector v = ColumnVector.datesFromBoxedLongs(values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampParams")
  void testTimestamp(ReductionOp op, Long[] values, TimeUnit timeUnit, Scalar expected) {
    try (ColumnVector v = ColumnVector.timestampsFromBoxedLongs(timeUnit, values)) {
      Scalar result = v.reduce(op, expected.type);
      assertEquals(expected, result);
    }
  }
}
