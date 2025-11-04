/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import ai.rapids.cudf.DType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** A literal value in an AST expression. */
public final class Literal extends AstExpression {
  private final DType type;
  private final byte[] serializedValue;

  @Override
  public String toString() {
    return "Literal " + type + " " + java.util.Arrays.toString(serializedValue);
  }

  /** Construct a null literal of the specified type. */
  public static Literal ofNull(DType type) {
    return new Literal(type, null);
  }

  /** Construct a boolean literal with the specified value. */
  public static Literal ofBoolean(boolean value) {
    return new Literal(DType.BOOL8, new byte[] { value ? (byte) 1 : (byte) 0 });
  }

  /** Construct a boolean literal with the specified value or null. */
  public static Literal ofBoolean(Boolean value) {
    if (value == null) {
      return ofNull(DType.BOOL8);
    }
    return ofBoolean(value.booleanValue());
  }

  /** Construct a byte literal with the specified value. */
  public static Literal ofByte(byte value) {
    return new Literal(DType.INT8, new byte[] { value });
  }

  /** Construct a byte literal with the specified value or null. */
  public static Literal ofByte(Byte value) {
    if (value == null) {
      return ofNull(DType.INT8);
    }
    return ofByte(value.byteValue());
  }

  /** Construct a short literal with the specified value. */
  public static Literal ofShort(short value) {
    byte[] serializedValue = new byte[Short.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putShort(value);
    return new Literal(DType.INT16, serializedValue);
  }

  /** Construct a short literal with the specified value or null. */
  public static Literal ofShort(Short value) {
    if (value == null) {
      return ofNull(DType.INT16);
    }
    return ofShort(value.shortValue());
  }

  /** Construct an integer literal with the specified value. */
  public static Literal ofInt(int value) {
    return ofIntBasedType(DType.INT32, value);
  }

  /** Construct an integer literal with the specified value or null. */
  public static Literal ofInt(Integer value) {
    if (value == null) {
      return ofNull(DType.INT32);
    }
    return ofInt(value.intValue());
  }

  /** Construct a long literal with the specified value. */
  public static Literal ofLong(long value) {
    return ofLongBasedType(DType.INT64, value);
  }

  /** Construct a long literal with the specified value or null. */
  public static Literal ofLong(Long value) {
    if (value == null) {
      return ofNull(DType.INT64);
    }
    return ofLong(value.longValue());
  }

  /** Construct a float literal with the specified value. */
  public static Literal ofFloat(float value) {
    byte[] serializedValue = new byte[Float.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putFloat(value);
    return new Literal(DType.FLOAT32, serializedValue);
  }

  /** Construct a float literal with the specified value or null. */
  public static Literal ofFloat(Float value) {
    if (value == null) {
      return ofNull(DType.FLOAT32);
    }
    return ofFloat(value.floatValue());
  }

  /** Construct a double literal with the specified value. */
  public static Literal ofDouble(double value) {
    byte[] serializedValue = new byte[Double.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putDouble(value);
    return new Literal(DType.FLOAT64, serializedValue);
  }

  /** Construct a double literal with the specified value or null. */
  public static Literal ofDouble(Double value) {
    if (value == null) {
      return ofNull(DType.FLOAT64);
    }
    return ofDouble(value.doubleValue());
  }

  /** Construct a timestamp days literal with the specified value. */
  public static Literal ofTimestampDaysFromInt(int value) {
    return ofIntBasedType(DType.TIMESTAMP_DAYS, value);
  }

  /** Construct a timestamp days literal with the specified value or null. */
  public static Literal ofTimestampDaysFromInt(Integer value) {
    if (value == null) {
      return ofNull(DType.TIMESTAMP_DAYS);
    }
    return ofTimestampDaysFromInt(value.intValue());
  }

  /** Construct a long-based timestamp literal with the specified value. */
  public static Literal ofTimestampFromLong(DType type, long value) {
    if (!type.isTimestampType()) {
      throw new IllegalArgumentException("type is not a timestamp: " + type);
    }
    if (type.equals(DType.TIMESTAMP_DAYS)) {
      int intValue = (int)value;
      if (value != intValue) {
        throw new IllegalArgumentException("value too large for type " + type + ": " + value);
      }
      return ofTimestampDaysFromInt(intValue);
    }
    return ofLongBasedType(type, value);
  }

  /** Construct a long-based timestamp literal with the specified value or null. */
  public static Literal ofTimestampFromLong(DType type, Long value) {
    if (value == null) {
      return ofNull(type);
    }
    return ofTimestampFromLong(type, value.longValue());
  }

  /** Construct a duration days literal with the specified value. */
  public static Literal ofDurationDaysFromInt(int value) {
    return ofIntBasedType(DType.DURATION_DAYS, value);
  }

  /** Construct a duration days literal with the specified value or null. */
  public static Literal ofDurationDaysFromInt(Integer value) {
    if (value == null) {
      return ofNull(DType.DURATION_DAYS);
    }
    return ofDurationDaysFromInt(value.intValue());
  }

  /** Construct a long-based duration literal with the specified value. */
  public static Literal ofDurationFromLong(DType type, long value) {
    if (!type.isDurationType()) {
      throw new IllegalArgumentException("type is not a timestamp: " + type);
    }
    if (type.equals(DType.DURATION_DAYS)) {
      int intValue = (int)value;
      if (value != intValue) {
        throw new IllegalArgumentException("value too large for type " + type + ": " + value);
      }
      return ofDurationDaysFromInt(intValue);
    }
    return ofLongBasedType(type, value);
  }

  /** Construct a long-based duration literal with the specified value or null. */
  public static Literal ofDurationFromLong(DType type, Long value) {
    if (value == null) {
      return ofNull(type);
    }
    return ofDurationFromLong(type, value.longValue());
  }

  /** Construct a string literal with the specified value or null. */
  public static Literal ofString(String value) {
    if (value == null) {
      return ofNull(DType.STRING);
    }
    return ofUTF8String(value.getBytes(StandardCharsets.UTF_8));
  }

  /** Construct a string literal directly with byte array to skip transcoding. */
  public static Literal ofUTF8String(byte[] stringBytes) {
    if (stringBytes == null) {
      return ofNull(DType.STRING);
    }
    byte[] serializedValue = new byte[stringBytes.length + Integer.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putInt(stringBytes.length);
    System.arraycopy(stringBytes, 0, serializedValue, Integer.BYTES, stringBytes.length);
    return new Literal(DType.STRING, serializedValue);
  }

  Literal(DType type, byte[] serializedValue) {
    this.type = type;
    this.serializedValue = serializedValue;
  }

  @Override
  int getSerializedSize() {
    ExpressionType nodeType = serializedValue != null
        ? ExpressionType.VALID_LITERAL : ExpressionType.NULL_LITERAL;
    int size = nodeType.getSerializedSize() + getDataTypeSerializedSize();
    if (serializedValue != null) {
      size += serializedValue.length;
    }
    return size;
  }

  @Override
  void serialize(ByteBuffer bb) {
    ExpressionType nodeType = serializedValue != null
        ? ExpressionType.VALID_LITERAL : ExpressionType.NULL_LITERAL;
    nodeType.serialize(bb);
    serializeDataType(bb);
    if (serializedValue != null) {
      bb.put(serializedValue);
    }
  }

  private int getDataTypeSerializedSize() {
    int nativeTypeId = type.getTypeId().getNativeId();
    assert nativeTypeId == (byte) nativeTypeId : "Type ID does not fit in a byte";
    if (type.isDecimalType()) {
      assert type.getScale() == (byte) type.getScale() : "Decimal scale does not fit in a byte";
      return 2;
    }
    return 1;
  }

  private void serializeDataType(ByteBuffer bb) {
    byte nativeTypeId = (byte) type.getTypeId().getNativeId();
    assert nativeTypeId == type.getTypeId().getNativeId() : "DType ID does not fit in a byte";
    bb.put(nativeTypeId);
    if (type.isDecimalType()) {
      byte scale = (byte) type.getScale();
      assert scale == (byte) type.getScale() : "Decimal scale does not fit in a byte";
      bb.put(scale);
    }
  }

  private static Literal ofIntBasedType(DType type, int value) {
    assert type.getSizeInBytes() == Integer.BYTES;
    byte[] serializedValue = new byte[Integer.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putInt(value);
    return new Literal(type, serializedValue);
  }

  private static Literal ofLongBasedType(DType type, long value) {
    assert type.getSizeInBytes() == Long.BYTES;
    byte[] serializedValue = new byte[Long.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putLong(value);
    return new Literal(type, serializedValue);
  }
}
