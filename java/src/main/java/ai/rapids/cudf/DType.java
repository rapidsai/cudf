/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.cudf;

import java.math.BigDecimal;
import java.util.EnumSet;
import java.util.Objects;

public final class DType {

  public static final int DECIMAL32_MAX_PRECISION = 10;
  public static final int DECIMAL64_MAX_PRECISION = 19;

  /* enum representing various types. Whenever a new non-decimal type is added please make sure
  below sections are updated as well:
  1. Create a singleton object of the new type.
  2. Update SINGLETON_DTYPE_LOOKUP to reflect new type. The order should be maintained between
  DTypeEnum and SINGLETON_DTYPE_LOOKUP */
  public enum DTypeEnum {
    EMPTY(0, 0, "NOT SUPPORTED"),
    INT8(1, 1, "byte"),
    INT16(2, 2, "short"),
    INT32(4, 3, "int"),
    INT64(8, 4, "long"),
    UINT8(1, 5, "uint8"),
    UINT16(2, 6, "uint16"),
    UINT32(4, 7, "uint32"),
    UINT64(8, 8, "uint64"),
    FLOAT32(4, 9, "float"),
    FLOAT64(8, 10, "double"),
    /**
     * Byte wise true non-0/false 0.  In general true will be 1.
     */
    BOOL8(1, 11, "bool"),
    /**
     * Days since the UNIX epoch
     */
    TIMESTAMP_DAYS(4, 12, "date32"),
    /**
     * s since the UNIX epoch
     */
    TIMESTAMP_SECONDS(8, 13, "timestamp[s]"),
    /**
     * ms since the UNIX epoch
     */
    TIMESTAMP_MILLISECONDS(8, 14, "timestamp[ms]"),
    /**
     * microseconds since the UNIX epoch
     */
    TIMESTAMP_MICROSECONDS(8, 15, "timestamp[us]"),
    /**
     * ns since the UNIX epoch
     */
    TIMESTAMP_NANOSECONDS(8, 16, "timestamp[ns]"),

    //We currently don't have mappings for duration type to I/O files, and these
    //simpleNames might change in future when we do
    DURATION_DAYS(4, 17, "int32"),
    DURATION_SECONDS(8, 18, "int64"),
    DURATION_MILLISECONDS(8, 19, "int64"),
    DURATION_MICROSECONDS(8, 20, "int64"),
    DURATION_NANOSECONDS(8, 21, "int64"),
    //DICTIONARY32(4, 22, "NO IDEA"),

    STRING(0, 23, "str"),
    LIST(0, 24, "list"),
    DECIMAL32(4, 25, "decimal32"),
    DECIMAL64(8, 26, "decimal64"),
    STRUCT(0, 27, "struct");

    final int sizeInBytes;
    final int nativeId;
    final String simpleName;

    DTypeEnum(int sizeInBytes, int nativeId, String simpleName) {
      this.sizeInBytes = sizeInBytes;
      this.nativeId = nativeId;
      this.simpleName = simpleName;
    }

    public int getNativeId() { return nativeId; }
  }

  final DTypeEnum typeId;
  private final int scale;

  private DType(DTypeEnum id) {
    typeId = id;
    scale = 0;
  }

  /**
   * Constructor for Decimal Type
   * @param id Enum representing data type.
   * @param decimalScale Scale of fixed point decimal type
   */
  private DType(DTypeEnum id, int decimalScale) {
    typeId = id;
    scale = decimalScale;
  }

  public static final DType EMPTY = new DType(DTypeEnum.EMPTY);
  public static final DType INT8 = new DType(DTypeEnum.INT8);
  public static final DType INT16 = new DType(DTypeEnum.INT16);
  public static final DType INT32 = new DType(DTypeEnum.INT32);
  public static final DType INT64 = new DType(DTypeEnum.INT64);
  public static final DType UINT8 = new DType(DTypeEnum.UINT8);
  public static final DType UINT16 = new DType(DTypeEnum.UINT16);
  public static final DType UINT32 = new DType(DTypeEnum.UINT32);
  public static final DType UINT64 = new DType(DTypeEnum.UINT64);
  public static final DType FLOAT32 = new DType(DTypeEnum.FLOAT32);
  public static final DType FLOAT64 = new DType(DTypeEnum.FLOAT64);
  public static final DType BOOL8 = new DType(DTypeEnum.BOOL8);
  public static final DType TIMESTAMP_DAYS = new DType(DTypeEnum.TIMESTAMP_DAYS);
  public static final DType TIMESTAMP_SECONDS = new DType(DTypeEnum.TIMESTAMP_SECONDS);
  public static final DType TIMESTAMP_MILLISECONDS = new DType(DTypeEnum.TIMESTAMP_MILLISECONDS);
  public static final DType TIMESTAMP_MICROSECONDS = new DType(DTypeEnum.TIMESTAMP_MICROSECONDS);
  public static final DType TIMESTAMP_NANOSECONDS = new DType(DTypeEnum.TIMESTAMP_NANOSECONDS);
  public static final DType DURATION_DAYS = new DType(DTypeEnum.DURATION_DAYS);
  public static final DType DURATION_SECONDS = new DType(DTypeEnum.DURATION_SECONDS);
  public static final DType DURATION_MILLISECONDS = new DType(DTypeEnum.DURATION_MILLISECONDS);
  public static final DType DURATION_MICROSECONDS = new DType(DTypeEnum.DURATION_MICROSECONDS);
  public static final DType DURATION_NANOSECONDS = new DType(DTypeEnum.DURATION_NANOSECONDS);
  public static final DType STRING = new DType(DTypeEnum.STRING);
  public static final DType LIST = new DType(DTypeEnum.LIST);
  public static final DType STRUCT = new DType(DTypeEnum.STRUCT);

  /* This is used in fromNative method to return singleton object for non-decimal types.
  Please make sure the order here is same as that of DTypeEnum. Whenever a new non-decimal
  type is added in DTypeEnum, this array needs to be updated as well.*/
  private static final DType[] SINGLETON_DTYPE_LOOKUP = new DType[]{
      EMPTY,
      INT8,
      INT16,
      INT32,
      INT64,
      UINT8,
      UINT16,
      UINT32,
      UINT64,
      FLOAT32,
      FLOAT64,
      BOOL8,
      TIMESTAMP_DAYS,
      TIMESTAMP_SECONDS,
      TIMESTAMP_MILLISECONDS,
      TIMESTAMP_MICROSECONDS,
      TIMESTAMP_NANOSECONDS,
      DURATION_DAYS,
      DURATION_SECONDS,
      DURATION_MILLISECONDS,
      DURATION_MICROSECONDS,
      DURATION_NANOSECONDS,
      null, // DICTIONARY32
      STRING,
      LIST,
      null, // DECIMAL32
      null, // DECIMAL64
      STRUCT
  };

  /**
   * This only works for fixed width types. Variable width types like strings the value is
   * undefined and should be ignored.
   *
   * @return size of type in bytes.
   */
  public int getSizeInBytes() { return typeId.sizeInBytes; }

  /**
   * Returns scale for Decimal Type.
   * @return scale base-10 exponent to multiply the unscaled value to produce the decimal value.
   * Example: Consider unscaled value = 123456
   *         if scale = -2, decimal value = 123456 * 10^-2 = 1234.56
   *         if scale = 2, decimal value = 123456 * 10^2 = 12345600
   */
  public int getScale() { return scale; }

  /**
   * Returns string name mapped to type.
   * @return name corresponding to type
   */
  public String getSimpleName() { return typeId.simpleName; }

  /**
   * Return enum for this DType
   * @return DTypeEnum
   */
  public DTypeEnum getTypeId() {
    return typeId;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    DType type = (DType) o;
    return scale == type.scale && typeId == type.typeId;
  }

  @Override
  public int hashCode() {
    return Objects.hash(typeId, scale);
  }

  @Override
  public String toString() {
    if (isDecimalType()) {
      return typeId + " scale:" + scale;
    } else {
      return String.valueOf(typeId);
    }
  }

  /**
   * Factory method for non-decimal DType instances.
   * @param dt  enum corresponding to datatype.
   * @return DType
   */
  public static DType create(DTypeEnum dt) {
    if (dt == DTypeEnum.DECIMAL32 || dt == DTypeEnum.DECIMAL64) {
      throw new IllegalArgumentException("Could not create a Decimal DType without scale");
    }
    return DType.fromNative(dt.nativeId, 0);
  }

  /**
   * Factory method specialized for decimal DType instances.
   * @param dt  enum corresponding to datatype.
   * @param scale base-10 exponent to multiply the unscaled value to produce the decimal value.
   * Example: Consider unscaled value = 123456
   *         if scale = -2, decimal value = 123456 * 10^-2 = 1234.56
   *         if scale = 2, decimal value = 123456 * 10^2 = 12345600
   * @return DType
   */
  public static DType create(DTypeEnum dt, int scale) {
    if (dt != DTypeEnum.DECIMAL32 && dt != DTypeEnum.DECIMAL64) {
      throw new IllegalArgumentException("Could not create a non-Decimal DType with scale");
    }
    return DType.fromNative(dt.nativeId, scale);
  }

  /**
   * Factory method for DType instances
   * @param nativeId nativeId of DataTypeEnun
   * @param scale base-10 exponent to multiply the unscaled value to produce the decimal value
   * Example: Consider unscaled value = 123456
   *         if scale = -2, decimal value = 123456 * 10^-2 = 1234.56
   *         if scale = 2, decimal value = 123456 * 10^2 = 12345600
   * @return DType
   */
  public static DType fromNative(int nativeId, int scale) {
    if (nativeId >=0 && nativeId < SINGLETON_DTYPE_LOOKUP.length) {
      DType ret = SINGLETON_DTYPE_LOOKUP[nativeId];
      if (ret != null) {
        assert ret.typeId.nativeId == nativeId : "Something went wrong and it looks like " +
          "SINGLETON_DTYPE_LOOKUP is out of sync";
        return ret;
      }
      if (nativeId == DTypeEnum.DECIMAL32.nativeId) {
        if (-scale > DECIMAL32_MAX_PRECISION) {
          throw new IllegalArgumentException(
              "Scale " + (-scale) + " exceeds DECIMAL32_MAX_PRECISION " + DECIMAL32_MAX_PRECISION);
        }
        return new DType(DTypeEnum.DECIMAL32, scale);
      }
      if (nativeId == DTypeEnum.DECIMAL64.nativeId) {
        if (-scale > DECIMAL64_MAX_PRECISION) {
          throw new IllegalArgumentException(
              "Scale " + (-scale) + " exceeds DECIMAL64_MAX_PRECISION " + DECIMAL64_MAX_PRECISION);
        }
        return new DType(DTypeEnum.DECIMAL64, scale);
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a DType");
  }

  /**
   * Create decimal-like DType using precision and scale of Java BigDecimal.
   *
   * @param dec BigDecimal
   * @return DType
   */
  public static DType fromJavaBigDecimal(BigDecimal dec) {
    // Notice: Compared to scale of Java BigDecimal, scale of libcudf works in opposite.
    // So, we negate the scale value before passing it into constructor.
    if (dec.precision() <= DECIMAL32_MAX_PRECISION) {
      return new DType(DTypeEnum.DECIMAL32, -dec.scale());
    } else if (dec.precision() <= DECIMAL64_MAX_PRECISION) {
      return new DType(DTypeEnum.DECIMAL64, -dec.scale());
    }
    throw new IllegalArgumentException("Precision " + dec.precision() +
        " exceeds max precision cuDF can support " + DECIMAL64_MAX_PRECISION);
  }

  /**
   * Returns true for timestamps with time level resolution, as opposed to day level resolution
   */
  public boolean hasTimeResolution() {
    return TIME_RESOLUTION.contains(this.typeId);
  }

  /**
   * Returns true if this type is backed by int type
   * Namely this method will return true for the following types
   *       DType.INT32,
   *       DType.UINT32,
   *       DType.DURATION_DAYS,
   *       DType.TIMESTAMP_DAYS
   */
  public boolean isBackedByInt() {
    return INTS.contains(this.typeId);
  }

  /**
   * Returns true if this type is backed by long type
   * Namely this method will return true for the following types
   *       DType.INT64,
   *       DType.UINT64,
   *       DType.DURATION_SECONDS,
   *       DType.DURATION_MILLISECONDS,
   *       DType.DURATION_MICROSECONDS,
   *       DType.DURATION_NANOSECONDS,
   *       DType.TIMESTAMP_SECONDS,
   *       DType.TIMESTAMP_MILLISECONDS,
   *       DType.TIMESTAMP_MICROSECONDS,
   *       DType.TIMESTAMP_NANOSECONDS
   */
  public boolean isBackedByLong() {
    return LONGS.contains(this.typeId);
  }

  /**
   * Returns true if this type is backed by short type
   * Namely this method will return true for the following types
   *       DType.INT16,
   *       DType.UINT16
   */
  public boolean isBackedByShort() { return SHORTS.contains(this.typeId); }

  /**
   * Returns true if this type is backed by byte type
   * Namely this method will return true for the following types
   *       DType.INT8,
   *       DType.UINT8,
   *       DType.BOOL8
   */
  public boolean isBackedByByte() { return BYTES.contains(this.typeId); }

  /**
   * Returns true if this type is of decimal type
   * Namely this method will return true for the following types
   *       DType.DECIMAL32,
   *       DType.DECIMAL64
   */
  public boolean isDecimalType() { return DECIMALS.contains(this.typeId); }

  /**
   * Returns true for duration types
   */
  public boolean isDurationType() {
    return DURATION_TYPE.contains(this.typeId);
  }

  /**
   * Returns true for nested types
   */
  public boolean isNestedType() {
    return NESTED_TYPE.contains(this.typeId);
  }

  @Deprecated
  public boolean isTimestamp() {
    return TIMESTAMPS.contains(this.typeId);
  }

  public boolean isTimestampType() {
    return TIMESTAMPS.contains(this.typeId);
  }

  private static final EnumSet<DTypeEnum> TIMESTAMPS = EnumSet.of(
      DTypeEnum.TIMESTAMP_DAYS,
      DTypeEnum.TIMESTAMP_SECONDS,
      DTypeEnum.TIMESTAMP_MILLISECONDS,
      DTypeEnum.TIMESTAMP_MICROSECONDS,
      DTypeEnum.TIMESTAMP_NANOSECONDS);

  private static final EnumSet<DTypeEnum> TIME_RESOLUTION = EnumSet.of(
      DTypeEnum.TIMESTAMP_SECONDS,
      DTypeEnum.TIMESTAMP_MILLISECONDS,
      DTypeEnum.TIMESTAMP_MICROSECONDS,
      DTypeEnum.TIMESTAMP_NANOSECONDS);

  private static final EnumSet<DTypeEnum> DURATION_TYPE = EnumSet.of(
      DTypeEnum.DURATION_DAYS,
      DTypeEnum.DURATION_MICROSECONDS,
      DTypeEnum.DURATION_MILLISECONDS,
      DTypeEnum.DURATION_NANOSECONDS,
      DTypeEnum.DURATION_SECONDS
  );

  private static final EnumSet<DTypeEnum> LONGS = EnumSet.of(
      DTypeEnum.INT64,
      DTypeEnum.UINT64,
      DTypeEnum.DURATION_SECONDS,
      DTypeEnum.DURATION_MILLISECONDS,
      DTypeEnum.DURATION_MICROSECONDS,
      DTypeEnum.DURATION_NANOSECONDS,
      DTypeEnum.TIMESTAMP_SECONDS,
      DTypeEnum.TIMESTAMP_MILLISECONDS,
      DTypeEnum.TIMESTAMP_MICROSECONDS,
      DTypeEnum.TIMESTAMP_NANOSECONDS
  );

  private static final EnumSet<DTypeEnum> INTS = EnumSet.of(
      DTypeEnum.INT32,
      DTypeEnum.UINT32,
      DTypeEnum.DURATION_DAYS,
      DTypeEnum.TIMESTAMP_DAYS
  );

  private static final EnumSet<DTypeEnum> SHORTS = EnumSet.of(
      DTypeEnum.INT16,
      DTypeEnum.UINT16
  );

  private static final EnumSet<DTypeEnum> BYTES = EnumSet.of(
      DTypeEnum.INT8,
      DTypeEnum.UINT8,
      DTypeEnum.BOOL8
  );

  private static final EnumSet<DTypeEnum> DECIMALS = EnumSet.of(
      DTypeEnum.DECIMAL32,
      DTypeEnum.DECIMAL64
  );

  private static final EnumSet<DTypeEnum> NESTED_TYPE = EnumSet.of(
      DTypeEnum.LIST,
      DTypeEnum.STRUCT
  );
}