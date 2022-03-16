/*
 *
 *  Copyright (c) 2022, NVIDIA CORPORATION.
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

import java.math.BigDecimal;
import java.util.AbstractMap;
import java.util.Map;

public class DecimalUtils {

  /**
   * Creates a cuDF decimal type with precision and scale
   */
  public static DType createDecimalType(int precision, int scale) {
    if (precision <= DType.DECIMAL32_MAX_PRECISION) {
      return DType.create(DType.DTypeEnum.DECIMAL32, -scale);
    } else if (precision <= DType.DECIMAL64_MAX_PRECISION) {
      return DType.create(DType.DTypeEnum.DECIMAL64, -scale);
    } else if (precision <= DType.DECIMAL128_MAX_PRECISION) {
      return DType.create(DType.DTypeEnum.DECIMAL128, -scale);
    }
    throw new IllegalArgumentException("precision overflow: " + precision);
  }

  /**
   * Given decimal precision and scale, returns the lower and upper bound of current decimal type.
   *
   * Be very careful when comparing these CUDF decimal comparisons really only work
   * when both types are already the same precision and scale, and when you change the scale
   * you end up losing information.
   * @param precision the max precision of decimal type
   * @param scale the scale of decimal type
   * @return a Map Entry of BigDecimal, lower bound as the key, upper bound as the value
   */
  public static Map.Entry<BigDecimal, BigDecimal> bounds(int precision, int scale) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < precision; i++) sb.append("9");
    sb.append("e");
    sb.append(-scale);
    String boundStr = sb.toString();
    BigDecimal upperBound = new BigDecimal(boundStr);
    BigDecimal lowerBound = new BigDecimal("-" + boundStr);
    return new AbstractMap.SimpleImmutableEntry<>(lowerBound, upperBound);
  }

  /**
   * With precision and scale, checks each value of input decimal column for out of bound.
   * @return the boolean column represents whether specific values are out of bound or not
   */
  public static ColumnVector outOfBounds(ColumnView input, int precision, int scale) {
    Map.Entry<BigDecimal, BigDecimal> boundPair = bounds(precision, scale);
    BigDecimal lowerBound = boundPair.getKey();
    BigDecimal upperBound = boundPair.getValue();
    try (ColumnVector over = greaterThan(input, upperBound);
         ColumnVector under = lessThan(input, lowerBound)) {
      return over.or(under);
    }
  }

  /**
   * Because the native lessThan operator has issues with comparing decimal values that have different
   * precision and scale accurately. This method takes some special steps to get rid of these issues.
   */
  public static ColumnVector lessThan(ColumnView lhs, BigDecimal rhs) {
    assert (lhs.getType().isDecimalType());
    int leftScale = lhs.getType().getScale();
    int leftPrecision = lhs.getType().getDecimalMaxPrecision();

    // First we have to round the scalar (rhs) to the same scale as lhs.  Because this is a
    // less than and it is rhs that we are rounding, we will round away from 0 (UP)
    // to make sure we always return the correct value.
    // For example:
    //      100.1 < 100.19
    // If we rounded down the rhs 100.19 would become 100.1, and now 100.1 is not < 100.1
    BigDecimal roundedRhs = rhs.setScale(-leftScale, BigDecimal.ROUND_UP);

    if (roundedRhs.precision() > leftPrecision) {
      // converting rhs to the same precision as lhs would result in an overflow/error, but
      // the scale is the same so we can still figure this out. For example if LHS precision is
      // 4 and RHS precision is 5 we get the following...
      //  9999 <  99999 => true
      // -9999 <  99999 => true
      //  9999 < -99999 => false
      // -9999 < -99999 => false
      // so the result should be the same as RHS > 0
      try (Scalar isPositive = Scalar.fromBool(roundedRhs.compareTo(BigDecimal.ZERO) > 0)) {
        return ColumnVector.fromScalar(isPositive, (int) lhs.getRowCount());
      }
    }
    try (Scalar scalarRhs = Scalar.fromDecimal(roundedRhs.unscaledValue(), lhs.getType())) {
      return lhs.lessThan(scalarRhs);
    }
  }

  /**
   * Because the native lessThan operator has issues with comparing decimal values that have different
   * precision and scale accurately. This method takes some special steps to get rid of these issues.
   */
  public static ColumnVector lessThan(BinaryOperable lhs, BigDecimal rhs, int numRows) {
    if (lhs instanceof ColumnView) {
      return lessThan((ColumnView) lhs, rhs);
    }
    Scalar scalarLhs = (Scalar) lhs;
    if (scalarLhs.isValid()) {
      try (Scalar isLess = Scalar.fromBool(scalarLhs.getBigDecimal().compareTo(rhs) < 0)) {
        return ColumnVector.fromScalar(isLess, numRows);
      }
    }
    try (Scalar nullScalar = Scalar.fromNull(DType.BOOL8)) {
      return ColumnVector.fromScalar(nullScalar, numRows);
    }
  }

  /**
   * Because the native greaterThan operator has issues with comparing decimal values that have different
   * precision and scale accurately. This method takes some special steps to get rid of these issues.
   */
  public static ColumnVector greaterThan(ColumnView lhs, BigDecimal rhs) {
    assert (lhs.getType().isDecimalType());
    int cvScale = lhs.getType().getScale();
    int maxPrecision = lhs.getType().getDecimalMaxPrecision();

    // First we have to round the scalar (rhs) to the same scale as lhs.  Because this is a
    // greater than and it is rhs that we are rounding, we will round towards 0 (DOWN)
    // to make sure we always return the correct value.
    // For example:
    //      100.2 > 100.19
    // If we rounded up the rhs 100.19 would become 100.2, and now 100.2 is not > 100.2
    BigDecimal roundedRhs = rhs.setScale(-cvScale, BigDecimal.ROUND_DOWN);

    if (roundedRhs.precision() > maxPrecision) {
      // converting rhs to the same precision as lhs would result in an overflow/error, but
      // the scale is the same so we can still figure this out. For example if LHS precision is
      // 4 and RHS precision is 5 we get the following...
      //  9999 >  99999 => false
      // -9999 >  99999 => false
      //  9999 > -99999 => true
      // -9999 > -99999 => true
      // so the result should be the same as RHS < 0
      try (Scalar isNegative = Scalar.fromBool(roundedRhs.compareTo(BigDecimal.ZERO) < 0)) {
        return ColumnVector.fromScalar(isNegative, (int) lhs.getRowCount());
      }
    }
    try (Scalar scalarRhs = Scalar.fromDecimal(roundedRhs.unscaledValue(), lhs.getType())) {
      return lhs.greaterThan(scalarRhs);
    }
  }
}
