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
   * Currently TIMESTAMP and DATE64 are treated the same as INT64. DATE32 is treated the same
   * as INT32.  TimeUnit conversion is not taken into account when doing operations on these
   * objects. This may change in the future, but for now the types returned will be stripped of
   * time units.  You can still get the desired behavior by casting the things involved to the
   * same time units doing the math, and casting it to a result with the desired time units.
   * <p>
   * BOOL8 is treated like an INT8.  Math on boolean operations makes little sense.  If
   * you want to to stay as a BOOL8 you will need to explicitly specify the output type.
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
    if (a == DType.INT64 || b == DType.INT64 ||
        a == DType.DATE64 || b == DType.DATE64 ||
        a == DType.TIMESTAMP || b == DType.TIMESTAMP) {
      return DType.INT64;
    }
    if (a == DType.INT32 || b == DType.INT32 ||
        a == DType.DATE32 || b == DType.DATE32) {
      return DType.INT32;
    }
    if (a == DType.INT16 || b == DType.INT16) {
      return DType.INT16;
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
}
