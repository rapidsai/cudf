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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Objects;

/**
 * A single scalar value.
 */
public final class Scalar implements AutoCloseable, BinaryOperable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger LOG = LoggerFactory.getLogger(Scalar.class);

  private final DType type;
  private int refCount;
  private final OffHeapState offHeap;

  public static Scalar fromNull(DType type) {
    switch (type) {
    case EMPTY:
    case BOOL8:
      return new Scalar(type, makeBool8Scalar(false, false));
    case INT8:
      return new Scalar(type, makeInt8Scalar((byte)0, false));
    case UINT8:
      return new Scalar(type, makeUint8Scalar((byte)0, false));
    case INT16:
      return new Scalar(type, makeInt16Scalar((short)0, false));
    case UINT16:
      return new Scalar(type, makeUint16Scalar((short)0, false));
    case INT32:
      return new Scalar(type, makeInt32Scalar(0, false));
    case UINT32:
      return new Scalar(type, makeUint32Scalar(0, false));
    case TIMESTAMP_DAYS:
      return new Scalar(type, makeTimestampDaysScalar(0, false));
    case FLOAT32:
      return new Scalar(type, makeFloat32Scalar(0, false));
    case FLOAT64:
      return new Scalar(type, makeFloat64Scalar(0, false));
    case INT64:
      return new Scalar(type, makeInt64Scalar(0, false));
    case UINT64:
      return new Scalar(type, makeUint64Scalar(0, false));
    case TIMESTAMP_SECONDS:
    case TIMESTAMP_MILLISECONDS:
    case TIMESTAMP_MICROSECONDS:
    case TIMESTAMP_NANOSECONDS:
      return new Scalar(type, makeTimestampTimeScalar(type.nativeId, 0, false));
    case STRING:
      return new Scalar(type, makeStringScalar(null, false));
    case DURATION_DAYS:
      return new Scalar(type, makeDurationDaysScalar(0, false));
    case DURATION_MICROSECONDS:
    case DURATION_MILLISECONDS:
    case DURATION_NANOSECONDS:
    case DURATION_SECONDS:
      return new Scalar(type, makeDurationTimeScalar(type.nativeId, 0, false));
    default:
      throw new IllegalArgumentException("Unexpected type: " + type);
    }
  }

  public static Scalar fromBool(boolean value) {
    return new Scalar(DType.BOOL8, makeBool8Scalar(value, true));
  }

  public static Scalar fromBool(Boolean value) {
    if (value == null) {
      return Scalar.fromNull(DType.BOOL8);
    }
    return Scalar.fromBool(value.booleanValue());
  }

  public static Scalar fromByte(byte value) {
    return new Scalar(DType.INT8, makeInt8Scalar(value, true));
  }

  public static Scalar fromByte(Byte value) {
    if (value == null) {
      return Scalar.fromNull(DType.INT8);
    }
    return Scalar.fromByte(value.byteValue());
  }

  public static Scalar fromUnsignedByte(byte value) {
    return new Scalar(DType.UINT8, makeUint8Scalar(value, true));
  }

  public static Scalar fromUnsignedByte(Byte value) {
    if (value == null) {
      return Scalar.fromNull(DType.UINT8);
    }
    return Scalar.fromUnsignedByte(value.byteValue());
  }

  public static Scalar fromShort(short value) {
    return new Scalar(DType.INT16, makeInt16Scalar(value, true));
  }

  public static Scalar fromShort(Short value) {
    if (value == null) {
      return Scalar.fromNull(DType.INT16);
    }
    return Scalar.fromShort(value.shortValue());
  }

  public static Scalar fromUnsignedShort(short value) {
    return new Scalar(DType.UINT16, makeUint16Scalar(value, true));
  }

  public static Scalar fromUnsignedShort(Short value) {
    if (value == null) {
      return Scalar.fromNull(DType.UINT16);
    }
    return Scalar.fromUnsignedShort(value.shortValue());
  }

  /**
   * Returns a DURATION_DAYS scalar
   * @param value - days
   * @return - Scalar value
   */
  public static Scalar durationDaysFromInt(int value) {
    return new Scalar(DType.DURATION_DAYS, makeDurationDaysScalar(value, true));
  }

  /**
   * Returns a DURATION_DAYS scalar
   * @param value - days
   * @return - Scalar value
   */
  public static Scalar durationDaysFromInt(Integer value) {
    if (value == null) {
      return Scalar.fromNull(DType.DURATION_DAYS);
    }
    return Scalar.durationDaysFromInt(value.intValue());
  }

  public static Scalar fromInt(int value) {
    return new Scalar(DType.INT32, makeInt32Scalar(value, true));
  }

  public static Scalar fromInt(Integer value) {
    if (value == null) {
      return Scalar.fromNull(DType.INT32);
    }
    return Scalar.fromInt(value.intValue());
  }

  public static Scalar fromUnsignedInt(int value) {
    return new Scalar(DType.UINT32, makeUint32Scalar(value, true));
  }

  public static Scalar fromUnsignedInt(Integer value) {
    if (value == null) {
      return Scalar.fromNull(DType.UINT32);
    }
    return Scalar.fromUnsignedInt(value.intValue());
  }

  public static Scalar fromLong(long value) {
    return new Scalar(DType.INT64, makeInt64Scalar(value, true));
  }

  public static Scalar fromLong(Long value) {
    if (value == null) {
      return Scalar.fromNull(DType.INT64);
    }
    return Scalar.fromLong(value.longValue());
  }

  public static Scalar fromUnsignedLong(long value) {
    return new Scalar(DType.UINT64, makeUint64Scalar(value, true));
  }

  public static Scalar fromUnsignedLong(Long value) {
    if (value == null) {
      return Scalar.fromNull(DType.UINT64);
    }
    return Scalar.fromUnsignedLong(value.longValue());
  }

  public static Scalar fromFloat(float value) {
    return new Scalar(DType.FLOAT32, makeFloat32Scalar(value, true));
  }

  public static Scalar fromFloat(Float value) {
    if (value == null) {
      return Scalar.fromNull(DType.FLOAT32);
    }
    return Scalar.fromFloat(value.floatValue());
  }

  public static Scalar fromDouble(double value) {
    return new Scalar(DType.FLOAT64, makeFloat64Scalar(value, true));
  }

  public static Scalar fromDouble(Double value) {
    if (value == null) {
      return Scalar.fromNull(DType.FLOAT64);
    }
    return Scalar.fromDouble(value.doubleValue());
  }

  public static Scalar timestampDaysFromInt(int value) {
    return new Scalar(DType.TIMESTAMP_DAYS, makeTimestampDaysScalar(value, true));
  }

  public static Scalar timestampDaysFromInt(Integer value) {
    if (value == null) {
      return Scalar.fromNull(DType.TIMESTAMP_DAYS);
    }
    return Scalar.timestampDaysFromInt(value.intValue());
  }

  /**
   * Returns a duration scalar based on the type parameter.
   * @param type - dtype of scalar to be returned
   * @param value - corresponding value for the scalar
   * @return - Scalar of the respective type
   */
  public static Scalar durationFromLong(DType type, long value) {
    if (type.isDurationType()) {
      if (type == DType.DURATION_DAYS) {
        int intValue = (int)value;
        if (value != intValue) {
          throw new IllegalArgumentException("value too large for type " + type + ": " + value);
        }
        return durationDaysFromInt(intValue);
      } else {
        return new Scalar(type, makeDurationTimeScalar(type.nativeId, value, true));
      }
    } else {
      throw new IllegalArgumentException("type is not a timestamp: " + type);
    }
  }

  /**
   * Returns a duration scalar based on the type parameter.
   * @param type - dtype of scalar to be returned
   * @param value - corresponding value for the scalar
   * @return - Scalar of the respective type
   */
  public static Scalar durationFromLong(DType type, Long value) {
    if (value == null) {
      return Scalar.fromNull(type);
    }
    return Scalar.durationFromLong(type, value.longValue());
  }

  public static Scalar timestampFromLong(DType type, long value) {
    if (type.isTimestamp()) {
      if (type == DType.TIMESTAMP_DAYS) {
        int intValue = (int)value;
        if (value != intValue) {
          throw new IllegalArgumentException("value too large for type " + type + ": " + value);
        }
        return timestampDaysFromInt(intValue);
      } else {
        return new Scalar(type, makeTimestampTimeScalar(type.nativeId, value, true));
      }
    } else {
      throw new IllegalArgumentException("type is not a timestamp: " + type);
    }
  }

  public static Scalar timestampFromLong(DType type, Long value) {
    if (value == null) {
      return Scalar.fromNull(type);
    }
    return Scalar.timestampFromLong(type, value.longValue());
  }

  public static Scalar fromString(String value) {
    if (value == null) {
      return fromNull(DType.STRING);
    }
    return new Scalar(DType.STRING, makeStringScalar(value.getBytes(StandardCharsets.UTF_8), true));
  }

  private static native void closeScalar(long scalarHandle);
  private static native boolean isScalarValid(long scalarHandle);
  private static native byte getByte(long scalarHandle);
  private static native short getShort(long scalarHandle);
  private static native int getInt(long scalarHandle);
  private static native long getLong(long scalarHandle);
  private static native float getFloat(long scalarHandle);
  private static native double getDouble(long scalarHandle);
  private static native byte[] getUTF8(long scalarHandle);
  private static native long makeBool8Scalar(boolean isValid, boolean value);
  private static native long makeInt8Scalar(byte value, boolean isValid);
  private static native long makeUint8Scalar(byte value, boolean isValid);
  private static native long makeInt16Scalar(short value, boolean isValid);
  private static native long makeUint16Scalar(short value, boolean isValid);
  private static native long makeInt32Scalar(int value, boolean isValid);
  private static native long makeUint32Scalar(int value, boolean isValid);
  private static native long makeInt64Scalar(long value, boolean isValid);
  private static native long makeUint64Scalar(long value, boolean isValid);
  private static native long makeFloat32Scalar(float value, boolean isValid);
  private static native long makeFloat64Scalar(double value, boolean isValid);
  private static native long makeStringScalar(byte[] value, boolean isValid);
  private static native long makeDurationDaysScalar(int value, boolean isValid);
  private static native long makeDurationTimeScalar(int dtype, long value, boolean isValid);
  private static native long makeTimestampDaysScalar(int value, boolean isValid);
  private static native long makeTimestampTimeScalar(int dtypeNativeId, long value, boolean isValid);


  Scalar(DType type, long scalarHandle) {
    this.type = type;
    this.offHeap = new OffHeapState(scalarHandle);
    incRefCount();
  }

  /**
   * Increment the reference count for this scalar.  You need to call close on this
   * to decrement the reference count again.
   */
  public synchronized Scalar incRefCount() {
    if (offHeap.scalarHandle == 0) {
      offHeap.logRefCountDebug("INC AFTER CLOSE " + this);
      throw new IllegalStateException("Scalar is already closed");
    }
    ++refCount;
    return this;
  }

  long getScalarHandle() {
    return offHeap.scalarHandle;
  }

  /**
   * Free the memory associated with a scalar.
   */
  @Override
  public synchronized void close() {
    refCount--;
    offHeap.delRef();
    if (refCount == 0) {
      offHeap.clean(false);
    } else if (refCount < 0) {
      offHeap.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times " + this);
    }
  }

  @Override
  public DType getType() {
    return type;
  }

  public boolean isValid() {
    return isScalarValid(getScalarHandle());
  }

  /**
   * Returns the scalar value as a boolean.
   */
  public boolean getBoolean() {
    return getByte(getScalarHandle()) != 0;
  }

  /**
   * Returns the scalar value as a byte.
   */
  public byte getByte() {
    return getByte(getScalarHandle());
  }

  /**
   * Returns the scalar value as a short.
   */
  public short getShort() {
    return getShort(getScalarHandle());
  }

  /**
   * Returns the scalar value as an int.
   */
  public int getInt() {
    return getInt(getScalarHandle());
  }

  /**
   * Returns the scalar value as a long.
   */
  public long getLong() {
    return getLong(getScalarHandle());
  }

  /**
   * Returns the scalar value as a float.
   */
  public float getFloat() {
    return getFloat(getScalarHandle());
  }

  /**
   * Returns the scalar value as a double.
   */
  public double getDouble() {
    return getDouble(getScalarHandle());
  }

  /**
   * Returns the scalar value as a Java string.
   */
  public String getJavaString() {
    return new String(getUTF8(getScalarHandle()), StandardCharsets.UTF_8);
  }

  /**
   * Returns the scalar value as UTF-8 data.
   */
  public byte[] getUTF8() {
    return getUTF8(getScalarHandle());
  }

  @Override
  public ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
    if (rhs instanceof ColumnVector) {
      ColumnVector cvRhs = (ColumnVector) rhs;
      return new ColumnVector(binaryOp(this, cvRhs, op, outType));
    } else {
      throw new IllegalArgumentException(rhs.getClass() + " is not supported as a binary op with " +
          "Scalar");
    }
  }

  static long binaryOp(Scalar lhs, ColumnVector rhs, BinaryOp op, DType outputType) {
    return binaryOpSV(lhs.getScalarHandle(), rhs.getNativeView(),
        op.nativeId, outputType.nativeId);
  }

  private static native long binaryOpSV(long lhs, long rhs, int op, int dtype);

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Scalar other = (Scalar) o;
    if (type != other.type) return false;
    boolean valid = isValid();
    if (valid != other.isValid()) return false;
    if (!valid) return true;
    switch (type) {
    case EMPTY:
      return true;
    case BOOL8:
      return getBoolean() == other.getBoolean();
    case INT8:
    case UINT8:
      return getByte() == other.getByte();
    case INT16:
    case UINT16:
      return getShort() == other.getShort();
    case INT32:
    case UINT32:
    case TIMESTAMP_DAYS:
      return getInt() == other.getInt();
    case FLOAT32:
      return getFloat() == other.getFloat();
    case FLOAT64:
      return getDouble() == other.getDouble();
    case INT64:
    case UINT64:
    case TIMESTAMP_SECONDS:
    case TIMESTAMP_MILLISECONDS:
    case TIMESTAMP_MICROSECONDS:
    case TIMESTAMP_NANOSECONDS:
      return getLong() == getLong();
    case STRING:
      return Arrays.equals(getUTF8(), other.getUTF8());
    default:
      throw new IllegalStateException("Unexpected type: " + type);
    }
  }

  @Override
  public int hashCode() {
    int valueHash = 0;
    if (isValid()) {
      switch (type) {
      case EMPTY:
        valueHash = 0;
        break;
      case BOOL8:
        valueHash = getBoolean() ? 1 : 0;
        break;
      case INT8:
      case UINT8:
        valueHash = getByte();
        break;
      case INT16:
      case UINT16:
        valueHash = getShort();
        break;
      case INT32:
      case UINT32:
      case TIMESTAMP_DAYS:
        valueHash = getInt();
        break;
      case INT64:
      case UINT64:
      case TIMESTAMP_SECONDS:
      case TIMESTAMP_MILLISECONDS:
      case TIMESTAMP_MICROSECONDS:
      case TIMESTAMP_NANOSECONDS:
        valueHash = Long.hashCode(getLong());
        break;
      case FLOAT32:
        valueHash = Float.hashCode(getFloat());
        break;
      case FLOAT64:
        valueHash = Double.hashCode(getDouble());
        break;
      case STRING:
        valueHash = Arrays.hashCode(getUTF8());
        break;
      default:
        throw new IllegalStateException("Unknown scalar type: " + type);
      }
    }
    return Objects.hash(type, valueHash);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("Scalar{type=");
    sb.append(type);
    if (getScalarHandle() != 0) {
      sb.append(" value=");
      switch (type) {
      case BOOL8:
        sb.append(getBoolean());
        break;
      case INT8:
        sb.append(getByte());
        break;
      case UINT8:
        sb.append(Byte.toUnsignedInt(getByte()));
        break;
      case INT16:
        sb.append(getShort());
        break;
      case UINT16:
        sb.append(Short.toUnsignedInt(getShort()));
        break;
      case INT32:
      case TIMESTAMP_DAYS:
        sb.append(getInt());
        break;
      case UINT32:
        sb.append(Integer.toUnsignedLong(getInt()));
        break;
      case INT64:
      case TIMESTAMP_SECONDS:
      case TIMESTAMP_MILLISECONDS:
      case TIMESTAMP_MICROSECONDS:
      case TIMESTAMP_NANOSECONDS:
        sb.append(getLong());
        break;
      case UINT64:
        sb.append(Long.toUnsignedString(getLong()));
        break;
      case FLOAT32:
        sb.append(getFloat());
        break;
      case FLOAT64:
        sb.append(getDouble());
        break;
      case STRING:
        sb.append('"');
        sb.append(getJavaString());
        sb.append('"');
        break;
      default:
        throw new IllegalArgumentException("Unknown scalar type: " + type);
      }
    }

    sb.append("} (ID: ");
    sb.append(offHeap.id);
    sb.append(" ");
    sb.append(Long.toHexString(offHeap.scalarHandle));
    sb.append(")");
    return sb.toString();
  }

  /**
   * Holds the off-heap state of the scalar so it can be cleaned up, even if it is leaked.
   */
  private static class OffHeapState extends MemoryCleaner.Cleaner {
    private long scalarHandle;

    OffHeapState(long scalarHandle) {
      this.scalarHandle = scalarHandle;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      if (scalarHandle != 0) {
        if (logErrorIfNotClean) {
          LOG.error("A SCALAR WAS LEAKED(ID: " + id + " " + Long.toHexString(scalarHandle) + ")");
          logRefCountDebug("Leaked scalar");
        }
        try {
          closeScalar(scalarHandle);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          scalarHandle = 0;
        }
        return true;
      }
      return false;
    }

    @Override
    public boolean isClean() {
      return scalarHandle == 0;
    }
  }
}
