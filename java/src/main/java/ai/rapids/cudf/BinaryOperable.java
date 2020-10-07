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

public interface BinaryOperable {
  /**
   * Finds the proper DType for an implicit output.  This follows the typical rules of
   * C++, Java, and most SQL implementations.
   * FLOAT64/double >
   * FLOAT32/float >
   * INT64/long >
   * INT32/int >
   * INT16/short >
   * INT8/byte/char
   * <p>
   * Currently most TIMESTAMPs are treated the same as INT64. TIMESTAMP_DAYS is treated the same
   * as INT32. All time information is stripped from them.  This may change in the future.
   * <p>
   * BOOL8 is treated like an INT8.  Math on boolean operations makes little sense.  If
   * you want to stay as a BOOL8 you will need to explicitly specify the output type.
   */
  static DType implicitConversion(BinaryOperable lhs, BinaryOperable rhs) {
    DType a = lhs.getType();
    DType b = rhs.getType();
    if (a == DType.FLOAT64 || b == DType.FLOAT64) {
      return DType.FLOAT64;
    }
    if (a == DType.FLOAT32 || b == DType.FLOAT32) {
      return DType.FLOAT32;
    }
    if (a == DType.UINT64 || b == DType.UINT64) {
      return DType.UINT64;
    }
    if (a == DType.INT64 || b == DType.INT64 ||
        a == DType.TIMESTAMP_MILLISECONDS || b == DType.TIMESTAMP_MILLISECONDS ||
        a == DType.TIMESTAMP_MICROSECONDS || b == DType.TIMESTAMP_MICROSECONDS ||
        a == DType.TIMESTAMP_SECONDS || b == DType.TIMESTAMP_SECONDS ||
        a == DType.TIMESTAMP_NANOSECONDS || b == DType.TIMESTAMP_NANOSECONDS) {
      return DType.INT64;
    }
    if (a == DType.UINT32 || b == DType.UINT32) {
      return DType.UINT32;
    }
    if (a == DType.INT32 || b == DType.INT32 ||
        a == DType.TIMESTAMP_DAYS || b == DType.TIMESTAMP_DAYS) {
      return DType.INT32;
    }
    if (a == DType.UINT16 || b == DType.UINT16) {
      return DType.UINT16;
    }
    if (a == DType.INT16 || b == DType.INT16) {
      return DType.INT16;
    }
    if (a == DType.UINT8 || b == DType.UINT8) {
      return DType.UINT8;
    }
    if (a == DType.INT8 || b == DType.INT8) {
      return DType.INT8;
    }
    if (a == DType.BOOL8 || b == DType.BOOL8) {
      return DType.BOOL8;
    }
    throw new IllegalArgumentException("Unsupported types " + a + " and " + b);
  }

  /**
   * Get the type of this data.
   */
  DType getType();

  /**
   * Multiple different binary operations.
   * @param op      the operation to perform
   * @param rhs     the rhs of the operation
   * @param outType the type of output you want.
   * @return the result
   */
  ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType);

  /**
   * Add + operator. this + rhs
   */
  default ColumnVector add(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.ADD, rhs, outType);
  }

  /**
   * Add + operator. this + rhs
   */
  default ColumnVector add(BinaryOperable rhs) {
    return add(rhs, implicitConversion(this, rhs));
  }

  /**
   * Subtract one vector from another with the given output type. this - rhs
   */
  default ColumnVector sub(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.SUB, rhs, outType);
  }

  /**
   * Subtract one vector from another. this - rhs
   */
  default ColumnVector sub(BinaryOperable rhs) {
    return sub(rhs, implicitConversion(this, rhs));
  }

  /**
   * Multiply two vectors together with the given output type. this * rhs
   */
  default ColumnVector mul(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.MUL, rhs, outType);
  }

  /**
   * Multiply two vectors together. this * rhs
   */
  default ColumnVector mul(BinaryOperable rhs) {
    return mul(rhs, implicitConversion(this, rhs));
  }

  /**
   * Divide one vector by another with the given output type. this / rhs
   */
  default ColumnVector div(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.DIV, rhs, outType);
  }

  /**
   * Divide one vector by another. this / rhs
   */
  default ColumnVector div(BinaryOperable rhs) {
    return div(rhs, implicitConversion(this, rhs));
  }

  /**
   * Divide one vector by another converting to FLOAT64 in between with the given output type.
   * (double)this / (double)rhs
   */
  default ColumnVector trueDiv(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.TRUE_DIV, rhs, outType);
  }

  /**
   * Divide one vector by another converting to FLOAT64 in between.
   * (double)this / (double)rhs
   */
  default ColumnVector trueDiv(BinaryOperable rhs) {
    return trueDiv(rhs, implicitConversion(this, rhs));
  }

  /**
   * Divide one vector by another and calculate the floor of the result with the given output type.
   * Math.floor(this/rhs)
   */
  default ColumnVector floorDiv(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.FLOOR_DIV, rhs, outType);
  }

  /**
   * Divide one vector by another and calculate the floor of the result.
   * Math.floor(this/rhs)
   */
  default ColumnVector floorDiv(BinaryOperable rhs) {
    return floorDiv(rhs, implicitConversion(this, rhs));
  }

  /**
   * Compute the modulus with the given output type.
   * this % rhs
   */
  default ColumnVector mod(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.MOD, rhs, outType);
  }

  /**
   * Compute the modulus.
   * this % rhs
   */
  default ColumnVector mod(BinaryOperable rhs) {
    return mod(rhs, implicitConversion(this, rhs));
  }

  /**
   * Compute the power with the given output type.
   * Math.pow(this, rhs)
   */
  default ColumnVector pow(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.POW, rhs, outType);
  }

  /**
   * Compute the power.
   * Math.pow(this, rhs)
   */
  default ColumnVector pow(BinaryOperable rhs) {
    return pow(rhs, implicitConversion(this, rhs));
  }

  /**
   * this == rhs 1 is true 0 is false with the output cast to the given type.
   */
  default ColumnVector equalTo(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.EQUAL, rhs, outType);
  }

  /**
   * this == rhs 1 is true 0 is false.  The output type is BOOL8.
   */
  default ColumnVector equalTo(BinaryOperable rhs) {
    return equalTo(rhs, DType.BOOL8);
  }

  /**
   * this != rhs 1 is true 0 is false with the output cast to the given type.
   */
  default ColumnVector notEqualTo(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.NOT_EQUAL, rhs, outType);
  }

  /**
   * this != rhs 1 is true 0 is false. The output type is BOOL8.
   */
  default ColumnVector notEqualTo(BinaryOperable rhs) {
    return notEqualTo(rhs, DType.BOOL8);
  }

  /**
   * this < rhs 1 is true 0 is false with the output cast to the given type.
   */
  default ColumnVector lessThan(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.LESS, rhs, outType);
  }

  /**
   * this < rhs 1 is true 0 is false.  The output type is BOOL8.
   */
  default ColumnVector lessThan(BinaryOperable rhs) {
    return lessThan(rhs, DType.BOOL8);
  }

  /**
   * this > rhs 1 is true 0 is false with the output cast to the given type.
   */
  default ColumnVector greaterThan(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.GREATER, rhs, outType);
  }

  /**
   * this > rhs 1 is true 0 is false.  The output type is BOOL8.
   */
  default ColumnVector greaterThan(BinaryOperable rhs) {
    return greaterThan(rhs, DType.BOOL8);
  }

  /**
   * this <= rhs 1 is true 0 is false with the output cast to the given type.
   */
  default ColumnVector lessOrEqualTo(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.LESS_EQUAL, rhs, outType);
  }

  /**
   * this <= rhs 1 is true 0 is false.  The output type is BOOL8.
   */
  default ColumnVector lessOrEqualTo(BinaryOperable rhs) {
    return lessOrEqualTo(rhs, DType.BOOL8);
  }

  /**
   * this >= rhs 1 is true 0 is false with the output cast to the given type.
   */
  default ColumnVector greaterOrEqualTo(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.GREATER_EQUAL, rhs, outType);
  }

  /**
   * this >= rhs 1 is true 0 is false.  The output type is BOOL8.
   */
  default ColumnVector greaterOrEqualTo(BinaryOperable rhs) {
    return greaterOrEqualTo(rhs, DType.BOOL8);
  }

  /**
   * Bit wise and (&) with the given output type. this & rhs
   */
  default ColumnVector bitAnd(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.BITWISE_AND, rhs, outType);
  }

  /**
   * Bit wise and (&). this & rhs
   */
  default ColumnVector bitAnd(BinaryOperable rhs) {
    return bitAnd(rhs, implicitConversion(this, rhs));
  }

  /**
   * Bit wise or (|) with the given output type. this | rhs
   */
  default ColumnVector bitOr(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.BITWISE_OR, rhs, outType);
  }

  /**
   * Bit wise or (|). this | rhs
   */
  default ColumnVector bitOr(BinaryOperable rhs) {
    return bitOr(rhs, implicitConversion(this, rhs));
  }

  /**
   * Bit wise xor (^) with the given output type. this ^ rhs
   */
  default ColumnVector bitXor(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.BITWISE_XOR, rhs, outType);
  }

  /**
   * Bit wise xor (^). this ^ rhs
   */
  default ColumnVector bitXor(BinaryOperable rhs) {
    return bitXor(rhs, implicitConversion(this, rhs));
  }

  /**
   * Logical and (&&) with the given output type. this && rhs
   */
  default ColumnVector and(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.LOGICAL_AND, rhs, outType);
  }

  /**
   * Logical and (&&). this && rhs
   */
  default ColumnVector and(BinaryOperable rhs) {
    return and(rhs, implicitConversion(this, rhs));
  }

  /**
   * Logical or (||) with the given output type. this || rhs  
   */
  default ColumnVector or(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.LOGICAL_OR, rhs, outType);
  }

  /**
   * Logical or (||). this || rhs
   */
  default ColumnVector or(BinaryOperable rhs) {
    return or(rhs, implicitConversion(this, rhs));
  }

  /**
   * Bitwise left shifts the values of this vector by shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] << shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar << shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] << shiftBy
   *
   */
  default ColumnVector shiftLeft(BinaryOperable shiftBy, DType outType) {
    return binaryOp(BinaryOp.SHIFT_LEFT, shiftBy, outType);
  }

  /**
   * Bitwise left shift the values of this vector by the shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] << shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar << shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] << shiftBy
   */
  default ColumnVector shiftLeft(BinaryOperable shiftBy) {
    return shiftLeft(shiftBy, implicitConversion(this, shiftBy));
  }

  /**
   * Bitwise right shift this vector by the shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] >> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >> shiftBy
   */
  default ColumnVector shiftRight(BinaryOperable shiftBy, DType outType) {
    return binaryOp(BinaryOp.SHIFT_RIGHT, shiftBy, outType);
  }

  /**
   * Bitwise right shift this vector by the shiftBy.
   *
   * If "this" and shiftBy are both vectors then, this[i] >> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >> shiftBy
   */
  default ColumnVector shiftRight(BinaryOperable shiftBy) {
    return shiftRight(shiftBy, implicitConversion(this, shiftBy));
  }

  /**
   * This method bitwise right shifts the values of this vector by the shiftBy.
   * This method always fills 0 irrespective of the sign of the number.
   *
   * If "this" and shiftBy are both vectors then, this[i] >>> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >>> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >>> shiftBy
   */
  default ColumnVector shiftRightUnsigned(BinaryOperable shiftBy, DType outType) {
    return binaryOp(BinaryOp.SHIFT_RIGHT_UNSIGNED, shiftBy, outType);
  }

  /**
   * This method bitwise right shifts the values of this vector by the shiftBy.
   * This method always fills 0 irrespective of the sign of the number.
   *
   * If "this" and shiftBy are both vectors then, this[i] >>> shiftBy[i]
   * If "this" is a scalar and shiftBy is a vector then returns a vector of size shiftBy.rows
   *    with the scalar >>> shiftBy[i]
   * If "this" is a vector and shiftBy is a scalar then returns a vector of size this.rows
   *    with this[i] >>> shiftBy
   */
  default ColumnVector shiftRightUnsigned(BinaryOperable shiftBy) {
    return shiftRightUnsigned(shiftBy, implicitConversion(this, shiftBy));
  }

  /**
   * Calculate the log with the specified base
   */
  default ColumnVector log(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.LOG_BASE, rhs, outType);
  }

  /**
   * Calculate the log with the specified base, output is the same as this.
   */
  default ColumnVector log(BinaryOperable rhs) {
    return log(rhs, getType());
  }

  /**
   * The function arctan2(y,x) or atan2(y,x) is defined as the angle in the Euclidean plane, given
   * in radians, between the positive x axis and the ray to the point (x, y) ≠ (0, 0).
   */
  default ColumnVector arctan2(BinaryOperable xCoordinate, DType outType) {
    return binaryOp(BinaryOp.ATAN2, xCoordinate, outType);
  }

  /**
   * The function arctan2(y,x) or atan2(y,x) is defined as the angle in the Euclidean plane, given
   * in radians, between the positive x axis and the ray to the point (x, y) ≠ (0, 0).
   */
  default ColumnVector arctan2(BinaryOperable xCoordinate) {
    return arctan2(xCoordinate, implicitConversion(this, xCoordinate));
  }

  /**
   * Returns the positive value of lhs mod rhs.
   *
   * r = lhs % rhs
   * if r < 0 then (r + rhs) % rhs
   * else r
   *
   */
  default ColumnVector pmod(BinaryOperable rhs, DType outputType) {
    return binaryOp(BinaryOp.PMOD, rhs, outputType);
  }

  /**
   * Returns the positive value of lhs mod rhs.
   *
   * r = lhs % rhs
   * if r < 0 then (r + rhs) % rhs
   * else r
   *
   */
  default ColumnVector pmod(BinaryOperable rhs) {
    return pmod(rhs, implicitConversion(this, rhs));
  }

  /**
   * like equalTo but NULL == NULL is TRUE and NULL == not NULL is FALSE
   */
  default ColumnVector equalToNullAware(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.NULL_EQUALS, rhs, outType);
  }

  /**
   * like equalTo but NULL == NULL is TRUE and NULL == not NULL is FALSE
   */
  default ColumnVector equalToNullAware(BinaryOperable rhs) {
    return equalToNullAware(rhs, DType.BOOL8);
  }

  /**
   * Returns the max non null value.
   */
  default ColumnVector maxNullAware(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.NULL_MAX, rhs, outType);
  }

  /**
   * Returns the max non null value.
   */
  default ColumnVector maxNullAware(BinaryOperable rhs) {
    return maxNullAware(rhs, implicitConversion(this, rhs));
  }

  /**
   * Returns the min non null value.
   */
  default ColumnVector minNullAware(BinaryOperable rhs, DType outType) {
    return binaryOp(BinaryOp.NULL_MIN, rhs, outType);
  }

  /**
   * Returns the min non null value.
   */
  default ColumnVector minNullAware(BinaryOperable rhs) {
    return minNullAware(rhs, implicitConversion(this, rhs));
  }
}
