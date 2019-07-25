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

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.Objects;

/**
 * A single scalar value.
 */
public final class Scalar implements BinaryOperable {
  /**
   * Generic NULL value.
   */
  public static final Scalar NULL = new Scalar(DType.INT8, TimeUnit.NONE);

  private static final EnumSet<DType> INTEGRAL_TYPES = EnumSet.of(
      DType.BOOL8, DType.INT8, DType.INT16, DType.INT32, DType.INT64,
      DType.DATE32, DType.DATE64, DType.TIMESTAMP);

  /*
   * In the native code all of the value are stored in a union with separate entries for each
   * DType.  Java has no equivalent to a union, so as a space saving compromise we store all
   * possible integer types (INT8 - INT64, DATE, TIMESTAMP, etc) in intTypeStorage.
   * Because conversion between a float and a double is not as cheap as it is for integers, we
   * split out float and double into floatTypeStorage and doubleTypeStorage.
   * String data is stored as UTF-8 bytes in stringTypeStorage.
   */
  final long intTypeStorage;
  final float floatTypeStorage;
  final double doubleTypeStorage;
  final byte[] stringTypeStorage;
  final DType type;
  final boolean isValid;
  // TimeUnit is not currently used by scalar values.  There are no operations that need it
  // When this changes we can support it.
  final TimeUnit timeUnit;

  private Scalar(long value, DType type, TimeUnit unit) {
    intTypeStorage = value;
    floatTypeStorage = 0;
    doubleTypeStorage = 0;
    stringTypeStorage = null;
    this.type = type;
    isValid = true;
    timeUnit = unit;
  }

  private Scalar(float value, DType type, TimeUnit unit) {
    intTypeStorage = 0;
    floatTypeStorage = value;
    doubleTypeStorage = 0;
    stringTypeStorage = null;
    this.type = type;
    isValid = true;
    timeUnit = unit;
  }

  private Scalar(double value, DType type, TimeUnit unit) {
    intTypeStorage = 0;
    floatTypeStorage = 0;
    doubleTypeStorage = value;
    stringTypeStorage = null;
    this.type = type;
    isValid = true;
    timeUnit = unit;
  }

  private Scalar(byte[] value, DType type, TimeUnit unit) {
    intTypeStorage = 0;
    floatTypeStorage = 0;
    doubleTypeStorage = 0;
    stringTypeStorage = value;
    this.type = type;
    isValid = value != null;
    timeUnit = unit;
  }

  private Scalar(DType type, TimeUnit unit) {
    intTypeStorage = 0;
    floatTypeStorage = 0;
    doubleTypeStorage = 0;
    stringTypeStorage = null;
    this.type = type;
    isValid = false;
    timeUnit = unit;
  }

  // These are invoked by native code to construct scalars.
  static Scalar fromNull(int dtype) {
    return new Scalar(DType.fromNative(dtype), TimeUnit.NONE);
  }

  static Scalar timestampFromNull(int nativeTimeUnit) {
    return timestampFromNull(TimeUnit.fromNative(nativeTimeUnit));
  }

  static Scalar timestampFromLong(long value, int nativeTimeUnit) {
    return timestampFromLong(value, TimeUnit.fromNative(nativeTimeUnit));
  }

  // These Scalar factory methods are called from native code.
  // If a new scalar type is supported then CudfJni also needs to be updated.

  public static Scalar fromNull(DType dtype) {
    return new Scalar(dtype, TimeUnit.NONE);
  }

  public static Scalar timestampFromNull(TimeUnit timeUnit) {
    return new Scalar(DType.TIMESTAMP, timeUnit);
  }

  public static Scalar fromBool(boolean value) {
    return new Scalar(value ? 1 : 0, DType.BOOL8, TimeUnit.NONE);
  }

  public static Scalar fromByte(byte value) {
    return new Scalar(value, DType.INT8, TimeUnit.NONE);
  }

  public static Scalar fromShort(short value) {
    return new Scalar(value, DType.INT16, TimeUnit.NONE);
  }

  public static Scalar fromInt(int value) {
    return new Scalar(value, DType.INT32, TimeUnit.NONE);
  }

  public static Scalar dateFromInt(int value) {
    return new Scalar(value, DType.DATE32, TimeUnit.NONE);
  }

  public static Scalar fromLong(long value) {
    return new Scalar(value, DType.INT64, TimeUnit.NONE);
  }

  public static Scalar dateFromLong(long value) {
    return new Scalar(value, DType.DATE64, TimeUnit.NONE);
  }

  public static Scalar timestampFromLong(long value) {
    return new Scalar(value, DType.TIMESTAMP, TimeUnit.MILLISECONDS);
  }

  public static Scalar timestampFromLong(long value, TimeUnit unit) {
    if (unit == TimeUnit.NONE) {
      unit = TimeUnit.MILLISECONDS;
    }
    return new Scalar(value, DType.TIMESTAMP, unit);
  }

  public static Scalar fromFloat(float value) {
    return new Scalar(value, DType.FLOAT32, TimeUnit.NONE);
  }

  public static Scalar fromDouble(double value) {
    return new Scalar(value, DType.FLOAT64, TimeUnit.NONE);
  }

  public static Scalar fromString(String value) {
    return new Scalar(value.getBytes(StandardCharsets.UTF_8), DType.STRING, TimeUnit.NONE);
  }

  public boolean isValid() {
    return isValid;
  }

  @Override
  public DType getType() {
    return type;
  }

  /**
   * Returns the scalar value as a boolean.
   */
  public boolean getBoolean() {
    if (INTEGRAL_TYPES.contains(type)) {
      return intTypeStorage != 0;
    } else if (type == DType.FLOAT32) {
      return floatTypeStorage != 0f;
    } else if (type == DType.FLOAT64) {
      return doubleTypeStorage != 0.;
    } else if (type == DType.STRING) {
      return Boolean.parseBoolean(getJavaString());
    }
    throw new IllegalStateException("Unexpected scalar type: " + type);
  }

  /**
   * Returns the scalar value as a byte.
   */
  public byte getByte() {
    if (INTEGRAL_TYPES.contains(type)) {
      return (byte) intTypeStorage;
    } else if (type == DType.FLOAT32) {
      return (byte) floatTypeStorage;
    } else if (type == DType.FLOAT64) {
      return (byte) doubleTypeStorage;
    } else if (type == DType.STRING) {
      return Byte.parseByte(getJavaString());
    }
    throw new IllegalStateException("Unexpected scalar type: " + type);
  }

  /**
   * Returns the scalar value as a short.
   */
  public short getShort() {
    if (INTEGRAL_TYPES.contains(type)) {
      return (short) intTypeStorage;
    } else if (type == DType.FLOAT32) {
      return (short) floatTypeStorage;
    } else if (type == DType.FLOAT64) {
      return (short) doubleTypeStorage;
    } else if (type == DType.STRING) {
      return Short.parseShort(getJavaString());
    }
    throw new IllegalStateException("Unexpected scalar type: " + type);
  }

  /**
   * Returns the scalar value as an int.
   */
  public int getInt() {
    if (INTEGRAL_TYPES.contains(type)) {
      return (int) intTypeStorage;
    } else if (type == DType.FLOAT32) {
      return (int) floatTypeStorage;
    } else if (type == DType.FLOAT64) {
      return (int) doubleTypeStorage;
    } else if (type == DType.STRING) {
      return Integer.parseInt(getJavaString());
    }
    throw new IllegalStateException("Unexpected scalar type: " + type);
  }

  /**
   * Returns the scalar value as a long.
   */
  public long getLong() {
    if (INTEGRAL_TYPES.contains(type)) {
      return intTypeStorage;
    } else if (type == DType.FLOAT32) {
      return (long) floatTypeStorage;
    } else if (type == DType.FLOAT64) {
      return (long) doubleTypeStorage;
    } else if (type == DType.STRING) {
      return Long.parseLong(getJavaString());
    }
    throw new IllegalStateException("Unexpected scalar type: " + type);
  }

  /**
   * Returns the scalar value as a float.
   */
  public float getFloat() {
    if (type == DType.FLOAT32) {
      return floatTypeStorage;
    } else if (type == DType.FLOAT64) {
      return (float) doubleTypeStorage;
    } else if (INTEGRAL_TYPES.contains(type)) {
      return intTypeStorage;
    } else if (type == DType.STRING) {
      return Float.parseFloat(getJavaString());
    }
    throw new IllegalStateException("Unexpected scalar type: " + type);
  }

  /**
   * Returns the scalar value as a double.
   */
  public double getDouble() {
    if (type == DType.FLOAT64) {
      return doubleTypeStorage;
    } else if (type == DType.FLOAT32) {
      return floatTypeStorage;
    } else if (INTEGRAL_TYPES.contains(type)) {
      return intTypeStorage;
    } else if (type == DType.STRING) {
      return Double.parseDouble(getJavaString());
    }
    throw new IllegalStateException("Unexpected scalar type: " + type);
  }

  /**
   * Returns the time unit associated with this scalar.
   */
  public TimeUnit getTimeUnit() {
    return timeUnit;
  }

  /**
   * Returns the scalar value as a Java string.
   */
  public String getJavaString() {
    if (type == DType.STRING) {
      return new String(stringTypeStorage, StandardCharsets.UTF_8);
    } else if (INTEGRAL_TYPES.contains(type)) {
      return Long.toString(intTypeStorage);
    } else if (type == DType.FLOAT32) {
      return Float.toString(floatTypeStorage);
    } else if (type == DType.FLOAT64) {
      return Double.toString(doubleTypeStorage);
    }
    throw new IllegalStateException("Unexpected scalar type: " + type);
  }

  /**
   * Returns the scalar value as UTF-8 data.
   */
  public byte[] getUTF8() {
    if (type == DType.STRING) {
      return stringTypeStorage;
    }
    return getJavaString().getBytes(StandardCharsets.UTF_8);
  }

  @Override
  public ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
    if (rhs instanceof ColumnVector) {
      ColumnVector cvRhs = (ColumnVector) rhs;
      return new ColumnVector(Cudf.gdfBinaryOp(this, cvRhs, op, outType));
    } else {
      throw new IllegalArgumentException(rhs.getClass() + " is not supported as a binary op with " +
          "Scalar");
    }
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Scalar scalar = (Scalar) o;
    return intTypeStorage == scalar.intTypeStorage &&
        Float.compare(scalar.floatTypeStorage, floatTypeStorage) == 0 &&
        Double.compare(scalar.doubleTypeStorage, doubleTypeStorage) == 0 &&
        isValid == scalar.isValid &&
        type == scalar.type &&
        timeUnit == scalar.timeUnit &&
        Arrays.equals(stringTypeStorage, scalar.stringTypeStorage);
  }

  @Override
  public int hashCode() {
    return Objects.hash(intTypeStorage, floatTypeStorage, doubleTypeStorage, stringTypeStorage,
        type, isValid, timeUnit);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("Scalar{type=");
    sb.append(type);
    sb.append(" value=");

    switch (type) {
      case BOOL8:
        sb.append(getBoolean());
        break;
      case INT8:
        sb.append(getByte());
        break;
      case INT16:
        sb.append(getShort());
        break;
      case INT32:
      case DATE32:
        sb.append(getInt());
        break;
      case INT64:
      case DATE64:
        sb.append(getLong());
        break;
      case FLOAT32:
        sb.append(getFloat());
        break;
      case FLOAT64:
        sb.append(getDouble());
        break;
      case TIMESTAMP:
        sb.append(getLong());
        sb.append(" unit=");
        sb.append(getTimeUnit());
        break;
      case STRING:
        sb.append('"');
        sb.append(getJavaString());
        sb.append('"');
        break;
      default:
        throw new IllegalArgumentException("Unknown scalar type: " + type);
    }

    sb.append("}");
    return sb.toString();
  }
}
