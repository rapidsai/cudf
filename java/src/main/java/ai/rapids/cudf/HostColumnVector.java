/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
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
import java.math.BigInteger;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.StringJoiner;
import java.util.function.Consumer;

/**
 * Similar to a ColumnVector, but the data is stored in host memory and accessible directly from
 * the JVM. This class holds references to off heap memory and is reference counted to know when
 * to release it.  Call close to decrement the reference count when you are done with the column,
 * and call incRefCount to increment the reference count.
 */
public final class HostColumnVector extends HostColumnVectorCore {
  /**
   * The size in bytes of an offset entry
   */
  static final int OFFSET_SIZE = DType.INT32.getSizeInBytes();

  private int refCount;

  /**
   * Create a new column vector with data populated on the host.
   */
  HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer) {
    this(type, rows, nullCount, hostDataBuffer, hostValidityBuffer, null);
  }

  /**
   * Create a new column vector with data populated on the host.
   * @param type               the type of the vector
   * @param rows               the number of rows in the vector.
   * @param nullCount          the number of nulls in the vector.
   * @param hostDataBuffer     The host side data for the vector. In the case of STRING
   *                           this is the string data stored as bytes.
   * @param hostValidityBuffer Arrow-like validity buffer 1 bit per row, with padding for
   *                           64-bit alignment.
   * @param offsetBuffer       only valid for STRING this is the offsets into
   *                           the hostDataBuffer indicating the start and end of a string
   *                           entry. It should be (rows + 1) ints.
   * @param nestedHcv          list of child HostColumnVectorCore(s) for complex types
   */

  //Constructor for lists and struct
  public HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
                   HostMemoryBuffer offsetBuffer, List<HostColumnVectorCore> nestedHcv) {
    super(type, rows, nullCount, hostDataBuffer, hostValidityBuffer, offsetBuffer, nestedHcv);
    refCount = 0;
    incRefCountInternal(true);
  }

  HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
                   HostMemoryBuffer offsetBuffer) {
    super(type, rows, nullCount, hostDataBuffer, hostValidityBuffer, offsetBuffer, new ArrayList<>());
    assert !type.equals(DType.LIST) : "This constructor should not be used for list type";
    if (nullCount.isPresent() && nullCount.get() > 0 && hostValidityBuffer == null) {
      throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
    }
    if (!type.equals(DType.STRING) && !type.equals(DType.LIST)) {
      assert offsetBuffer == null : "offsets are only supported for STRING and LIST";
    }
    refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * This is a really ugly API, but it is possible that the lifecycle of a column of
   * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
   * tracking code that this is expected for this column, and big scary warnings should
   * not be printed when this happens.
   */
  public void noWarnLeakExpected() {
    offHeap.noWarnLeakExpected();
  }

  /**
   * Close this Vector and free memory allocated for HostMemoryBuffer and DeviceMemoryBuffer
   */
  @Override
  public synchronized void close() {
    refCount--;
    offHeap.delRef();
    if (refCount == 0) {
      offHeap.clean(false);
      for( HostColumnVectorCore child : children) {
        child.close();
      }
    } else if (refCount < 0) {
      offHeap.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times " + this);
    }
  }

  @Override
  public String toString() {
    return "HostColumnVector{" +
        "rows=" + rows +
        ", type=" + type +
        ", nullCount=" + nullCount +
        ", offHeap=" + offHeap +
        '}';
  }

  /////////////////////////////////////////////////////////////////////////////
  // METADATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Increment the reference count for this column.  You need to call close on this
   * to decrement the reference count again.
   */
  public HostColumnVector incRefCount() {
    return incRefCountInternal(false);
  }

  private synchronized HostColumnVector incRefCountInternal(boolean isFirstTime) {
    offHeap.addRef();
    if (refCount <= 0 && !isFirstTime) {
      offHeap.logRefCountDebug("INC AFTER CLOSE " + this);
      throw new IllegalStateException("Column is already closed");
    }
    refCount++;
    return this;
  }

  /**
   * Returns this column's current refcount
   */
  synchronized int getRefCount() {
    return refCount;
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATA MOVEMENT
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Copy the data to the device.
   */
  public ColumnVector copyToDevice() {
    if (rows == 0) {
      if (type.isNestedType()) {
        return ColumnView.NestedColumnVector.createColumnVector(type, 0,
                null, null, null, Optional.of(0L), children);
      } else {
        return new ColumnVector(type, 0, Optional.of(0L), null, null, null);
      }
    }
    // The simplest way is just to copy the buffers and pass them down.
    DeviceMemoryBuffer data = null;
    DeviceMemoryBuffer valid = null;
    DeviceMemoryBuffer offsets = null;
    try {
      if (!type.isNestedType()) {
        HostMemoryBuffer hdata = this.offHeap.data;
        if (hdata != null) {
          long dataLen = rows * type.getSizeInBytes();
          if (type.equals(DType.STRING)) {
            // This needs a different type
            dataLen = getEndStringOffset(rows - 1);
            if (dataLen == 0 && getNullCount() == 0) {
              // This is a work around to an issue where a column of all empty strings must have at
              // least one byte or it will not be interpreted correctly.
              dataLen = 1;
            }
          }
          data = DeviceMemoryBuffer.allocate(dataLen);
          data.copyFromHostBuffer(hdata, 0, dataLen);
        }
        HostMemoryBuffer hvalid = this.offHeap.valid;
        if (hvalid != null) {
          long validLen = ColumnView.getNativeValidPointerSize((int) rows);
          valid = DeviceMemoryBuffer.allocate(validLen);
          valid.copyFromHostBuffer(hvalid, 0, validLen);
        }

        HostMemoryBuffer hoff = this.offHeap.offsets;
        if (hoff != null) {
          long offsetsLen = OFFSET_SIZE * (rows + 1);
          offsets = DeviceMemoryBuffer.allocate(offsetsLen);
          offsets.copyFromHostBuffer(hoff, 0, offsetsLen);
        }

        ColumnVector ret = new ColumnVector(type, rows, nullCount, data, valid, offsets);
        data = null;
        valid = null;
        offsets = null;
        return ret;
      } else {
        return ColumnView.NestedColumnVector.createColumnVector(
            type, (int) rows, offHeap.data, offHeap.valid, offHeap.offsets, nullCount, children);
      }
    } finally {
      if (data != null) {
        data.close();
      }
      if (valid != null) {
        valid.close();
      }
      if (offsets != null) {
        offsets.close();
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // BUILDER
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when
   * done with it. Please try to use {@see #build(int, Consumer)} instead to avoid needing to
   * close the builder.
   * @param type the type of vector to build.
   * @param rows the number of rows this builder can hold
   * @return the builder to use.
   */
  public static Builder builder(DType type, int rows) {
    return new Builder(type, rows, 0);
  }

  /**
   * Create a new Builder to hold the specified number of rows and with enough space to hold the
   * given amount of string data. Be sure to close the builder when done with it. Please try to
   * use {@see #build(int, int, Consumer)} instead to avoid needing to close the builder.
   * @param rows the number of rows this builder can hold
   * @param stringBufferSize the size of the string buffer to allocate.
   * @return the builder to use.
   */
  public static Builder builder(int rows, long stringBufferSize) {
    return new HostColumnVector.Builder(DType.STRING, rows, stringBufferSize);
  }

  /**
   * Create a new vector.
   * @param type       the type of vector to build.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static HostColumnVector build(DType type, int rows, Consumer<Builder> init) {
    try (HostColumnVector.Builder builder = builder(type, rows)) {
      init.accept(builder);
      return builder.build();
    }
  }

  public static HostColumnVector build(int rows, long stringBufferSize, Consumer<Builder> init) {
    try (HostColumnVector.Builder builder = builder(rows, stringBufferSize)) {
      init.accept(builder);
      return builder.build();
    }
  }

  public static<T> HostColumnVector fromLists(DataType dataType, List<T>... values) {
    try (ColumnBuilder cb = new ColumnBuilder(dataType, values.length)) {
      cb.appendLists(values);
      return cb.build();
    }
  }

  public static HostColumnVector fromStructs(DataType dataType,
                                             List<StructData> values) {
    try (ColumnBuilder cb = new ColumnBuilder(dataType, values.size())) {
      cb.appendStructValues(values);
      return cb.build();
    }
  }

  public static HostColumnVector fromStructs(DataType dataType, StructData... values) {
    try (ColumnBuilder cb = new ColumnBuilder(dataType, values.length)) {
      cb.appendStructValues(values);
      return cb.build();
    }
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector boolFromBytes(byte... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromBytes(byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned byte type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedBytes(byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromShorts(short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned short type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedShorts(short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationNanosecondsFromLongs(long... values) {
    return build(DType.DURATION_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationMicrosecondsFromLongs(long... values) {
    return build(DType.DURATION_MICROSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationMillisecondsFromLongs(long... values) {
    return build(DType.DURATION_MILLISECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationSecondsFromLongs(long... values) {
    return build(DType.DURATION_SECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationDaysFromInts(int... values) {
    return build(DType.DURATION_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromInts(int... values) {
    return build(DType.INT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned int type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedInts(int... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromLongs(long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned long type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedLongs(long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromFloats(float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromDoubles(double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector daysFromInts(int... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampMilliSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampMicroSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampNanoSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new decimal vector from unscaled values (int array) and scale.
   * The created vector is of type DType.DECIMAL32, whose max precision is 9.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static HostColumnVector decimalFromInts(int scale, int... values) {
    return build(DType.create(DType.DTypeEnum.DECIMAL32, scale), values.length, (b) -> b.appendUnscaledDecimalArray(values));
  }

  /**
   * Create a new decimal vector from unscaled values (long array) and scale.
   * The created vector is of type DType.DECIMAL64, whose max precision is 18.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static HostColumnVector decimalFromLongs(int scale, long... values) {
    return build(DType.create(DType.DTypeEnum.DECIMAL64, scale), values.length, (b) -> b.appendUnscaledDecimalArray(values));
  }

  /**
   * Create a new decimal vector from double floats with specific DecimalType and RoundingMode.
   * All doubles will be rescaled if necessary, according to scale of input DecimalType and RoundingMode.
   * If any overflow occurs in extracting integral part, an IllegalArgumentException will be thrown.
   * This API is inefficient because of slow double -> decimal conversion, so it is mainly for testing.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static HostColumnVector decimalFromDoubles(DType type, RoundingMode mode, double... values) {
    assert type.isDecimalType();
    if (type.typeId == DType.DTypeEnum.DECIMAL64) {
      long[] data = new long[values.length];
      for (int i = 0; i < values.length; i++) {
        BigDecimal dec = BigDecimal.valueOf(values[i]).setScale(-type.getScale(), mode);
        data[i] = dec.unscaledValue().longValueExact();
      }
      return build(type, values.length, (b) -> b.appendUnscaledDecimalArray(data));
    } else {
      int[] data = new int[values.length];
      for (int i = 0; i < values.length; i++) {
        BigDecimal dec = BigDecimal.valueOf(values[i]).setScale(-type.getScale(), mode);
        data[i] = dec.unscaledValue().intValueExact();
      }
      return build(type, values.length, (b) -> b.appendUnscaledDecimalArray(data));
    }
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls. This is really intended to be used only for testing as
   * it is slow and memory intensive to translate between java strings and UTF8 strings.
   */
  public static HostColumnVector fromStrings(String... values) {
    int rows = values.length;
    long nullCount = 0;
    // How many bytes do we need to hold the data.  Sorry this is really expensive
    long bufferSize = 0;
    for (String s: values) {
      if (s == null) {
        nullCount++;
      } else {
        bufferSize += s.getBytes(StandardCharsets.UTF_8).length;
      }
    }
    if (nullCount > 0) {
      return build(rows, bufferSize, (b) -> b.appendBoxed(values));
    }
    return build(rows, bufferSize, (b) -> {
      for (String s: values) {
        b.append(s);
      }
    });
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than building from primitive array of unscaledValues.
   * Notice:
   *  1. Input values will be rescaled with min scale (max scale in terms of java.math.BigDecimal),
   *  which avoids potential precision loss due to rounding. But there exists risk of precision overflow.
   *  2. The scale will be zero if all input values are null.
   */
  public static HostColumnVector fromDecimals(BigDecimal... values) {
    // 1. Fetch the element with max precision (maxDec). Fill with ZERO if inputs is empty.
    // 2. Fetch the max scale. Fill with ZERO if inputs is empty.
    // 3. Rescale the maxDec with the max scale, so to come out the max precision capacity we need.
    BigDecimal maxDec = Arrays.stream(values).filter(Objects::nonNull)
        .max(Comparator.comparingInt(BigDecimal::precision))
        .orElse(BigDecimal.ZERO);
    int maxScale = Arrays.stream(values).filter(Objects::nonNull)
        .map(decimal -> decimal.scale())
        .max(Comparator.naturalOrder())
        .orElse(0);
    maxDec = maxDec.setScale(maxScale, RoundingMode.UNNECESSARY);

    return build(DType.fromJavaBigDecimal(maxDec), values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedBooleans(Boolean... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedBytes(Byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned byte type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedBytes(Byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedShorts(Short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned short type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedShorts(Short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationNanosecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationMicrosecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationMillisecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_SECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationDaysFromBoxedInts(Integer... values) {
    return build(DType.DURATION_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedInts(Integer... values) {
    return build(DType.INT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned int type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedInts(Integer... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedLongs(Long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned long type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedLongs(Long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedFloats(Float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedDoubles(Double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampDaysFromBoxedInts(Integer... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampMilliSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampMicroSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampNanoSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Build
   */

  public static final class ColumnBuilder implements  AutoCloseable {

    private DType type;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private HostMemoryBuffer offsets;
    private long nullCount = 0l;
    //TODO nullable currently not used
    private boolean nullable;
    private long rows;
    private long estimatedRows;
    private boolean built = false;
    private List<ColumnBuilder> childBuilders = new ArrayList<>();

    private int currentIndex = 0;
    private int currentByteIndex = 0;


    public ColumnBuilder(HostColumnVector.DataType type, long estimatedRows) {
      this.type = type.getType();
      this.nullable = type.isNullable();
      this.rows = 0;
      this.estimatedRows = estimatedRows;
      for (int i = 0; i < type.getNumChildren(); i++) {
        childBuilders.add(new ColumnBuilder(type.getChild(i), estimatedRows));
      }
    }

    public HostColumnVector build() {
      List<HostColumnVectorCore> hostColumnVectorCoreList = new ArrayList<>();
      for (ColumnBuilder childBuilder : childBuilders) {
        hostColumnVectorCoreList.add(childBuilder.buildNestedInternal());
      }
      HostColumnVector hostColumnVector = new HostColumnVector(type, rows, Optional.of(nullCount), data, valid, offsets,
          hostColumnVectorCoreList);
      built = true;
      return hostColumnVector;
    }

    private HostColumnVectorCore buildNestedInternal() {
      List<HostColumnVectorCore> hostColumnVectorCoreList = new ArrayList<>();
      for (ColumnBuilder childBuilder : childBuilders) {
        hostColumnVectorCoreList.add(childBuilder.buildNestedInternal());
      }
      return new HostColumnVectorCore(type, rows, Optional.of(nullCount), data, valid, offsets, hostColumnVectorCoreList);
    }

    public ColumnBuilder appendLists(List... inputLists) {
      for (List inputList : inputLists) {
        // one row
        append(inputList);
      }
      return this;
    }

    public ColumnBuilder appendStructValues(List<StructData> inputList) {
      for (StructData structInput : inputList) {
        // one row
        append(structInput);
      }
      return this;
    }

    public ColumnBuilder appendStructValues(StructData... inputList) {
      for (StructData structInput : inputList) {
        append(structInput);
      }
      return this;
    }

    /**
     * A method that is responsible for growing the buffers as needed
     * and incrementing the row counts when we append values or nulls.
     * @param hasNull indicates whether the validity buffer needs to be considered, as the
     *                nullcount may not have been fully calculated yet
     * @param length used for strings
     */
    private void growBuffersAndRows(boolean hasNull, int length) {
      assert rows + 1 <= Integer.MAX_VALUE : "Row count cannot go over Integer.MAX_VALUE";
      rows++;
      long targetDataSize = 0;

      if (!type.isNestedType()) {
        if (type.equals(DType.STRING)) {
          targetDataSize = data == null ? length : currentByteIndex + length;
        } else {
          targetDataSize = data == null ? estimatedRows * type.getSizeInBytes() : rows * type.getSizeInBytes();
        }
      }

      if (targetDataSize > 0) {
        if (data == null) {
          data = HostMemoryBuffer.allocate(targetDataSize);
        } else {
          long maxLen;
          if (type.equals(DType.STRING)) {
            maxLen = Integer.MAX_VALUE;
          } else {
            maxLen = Integer.MAX_VALUE * (long) type.getSizeInBytes();
          }
          long oldLen = data.getLength();
          long newDataLen = Math.max(1, oldLen);
          while (targetDataSize > newDataLen) {
            newDataLen = newDataLen * 2;
          }
          if (newDataLen != oldLen) {
            newDataLen = Math.min(newDataLen, maxLen);
            if (newDataLen < targetDataSize) {
              throw new IllegalStateException("A data buffer for strings is not supported over 2GB in size");
            }
            HostMemoryBuffer newData = HostMemoryBuffer.allocate(newDataLen);
            data = copyBuffer(newData, data);
          }
        }
      }
      if (type.equals(DType.LIST) || type.equals(DType.STRING)) {
        if (offsets == null) {
          offsets = HostMemoryBuffer.allocate((estimatedRows + 1) * OFFSET_SIZE);
          offsets.setInt(0, 0);
        } else if ((rows +1) * OFFSET_SIZE > offsets.length) {
          long newOffsetLen = offsets.length * 2;
          HostMemoryBuffer newOffsets = HostMemoryBuffer.allocate(newOffsetLen);
          offsets = copyBuffer(newOffsets, offsets);
        }
      }
      if (hasNull || nullCount > 0) {
        if (valid == null) {
          long targetValidSize = ColumnView.getNativeValidPointerSize((int)estimatedRows);
          valid = HostMemoryBuffer.allocate(targetValidSize);
          valid.setMemory(0, targetValidSize, (byte) 0xFF);
        } else if (valid.length < ColumnView.getNativeValidPointerSize((int)rows)) {
          long newValidLen = valid.length * 2;
          HostMemoryBuffer newValid = HostMemoryBuffer.allocate(newValidLen);
          newValid.setMemory(0, newValidLen, (byte) 0xFF);
          valid = copyBuffer(newValid, valid);
        }
      }
    }

    private HostMemoryBuffer copyBuffer(HostMemoryBuffer targetBuffer, HostMemoryBuffer buffer) {
      try {
        targetBuffer.copyFromHostBuffer(0, buffer, 0, buffer.length);
        buffer.close();
        buffer = targetBuffer;
        targetBuffer = null;
      } finally {
        if (targetBuffer != null) {
          targetBuffer.close();
        }
      }
      return buffer;
    }

    /**
     * Method that sets the null bit in the validity vector
     * @param index the row index at which the null is marked
     */
    private void setNullAt(int index) {
      assert index < rows : "Index for null value should fit the column with " + rows + " rows";
      nullCount += BitVectorHelper.setNullAt(valid, index);
    }

    public final ColumnBuilder appendNull() {
      growBuffersAndRows(true, 0);
      setNullAt(currentIndex);
      currentIndex++;
      currentByteIndex += type.getSizeInBytes();
      if (type.hasOffsets()) {
        if (type.equals(DType.LIST)) {
          offsets.setInt(currentIndex * OFFSET_SIZE, childBuilders.get(0).getCurrentIndex());
        } else {
          // It is a String
          offsets.setInt(currentIndex * OFFSET_SIZE, currentByteIndex);
        }
      } else if (type.equals(DType.STRUCT)) {
        // structs propagate nulls to children and even further down if needed
        for (ColumnBuilder childBuilder : childBuilders) {
          childBuilder.appendNull();
        }
      }
      return this;
    }

    //For structs
    private ColumnBuilder append(StructData structData) {
      assert type.isNestedType();
      if (type.equals(DType.STRUCT)) {
        if (structData == null || structData.dataRecord == null) {
          return appendNull();
        } else {
          for (int i = 0; i < structData.getNumFields(); i++) {
            ColumnBuilder childBuilder = childBuilders.get(i);
            appendChildOrNull(childBuilder, structData.dataRecord.get(i));
          }
          endStruct();
        }
      }
      return this;
    }

    private boolean allChildrenHaveSameIndex() {
      if (childBuilders.size() > 0) {
        int expected = childBuilders.get(0).getCurrentIndex();
        for (ColumnBuilder child: childBuilders) {
          if (child.getCurrentIndex() != expected) {
            return false;
          }
        }
      }
      return true;
    }

    /**
     * If you want to build up a struct column you can get each child `builder.getChild(N)` and
     * append to all of them, then when you are done call `endStruct` to update this builder.
     * Do not start to append to the child and then append a null to this without ending the struct
     * first or you might not get the results that you expected.
     * @return this for chaining.
     */
    public ColumnBuilder endStruct() {
      assert type.equals(DType.STRUCT) : "This only works for structs";
      assert allChildrenHaveSameIndex() : "Appending structs data appears to be off " +
          childBuilders + " should all have the same currentIndex " + type;
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      currentIndex++;
      return this;
    }

    /**
     * If you want to build up a list column you can get `builder.getChild(0)` and append to than,
     * then when you are done call `endList` and everything that was appended to that builder
     * will now be in the next list. Do not start to append to the child and then append a null
     * to this without ending the list first or you might not get the results that you expected.
     * @return this for chaining.
     */
    public ColumnBuilder endList() {
      assert type.equals(DType.LIST);
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      currentIndex++;
      offsets.setInt(currentIndex * OFFSET_SIZE, childBuilders.get(0).getCurrentIndex());
      return this;
    }

    // For lists
    private <T> ColumnBuilder append(List<T> inputList) {
      if (inputList == null) {
        appendNull();
      } else {
        ColumnBuilder childBuilder = childBuilders.get(0);
        for (Object listElement : inputList) {
          appendChildOrNull(childBuilder, listElement);
        }
        endList();
      }
      return this;
    }

    private void appendChildOrNull(ColumnBuilder childBuilder, Object listElement) {
      if (listElement == null) {
        childBuilder.appendNull();
      } else if (listElement instanceof Integer) {
        childBuilder.append((Integer) listElement);
      } else if (listElement instanceof String) {
        childBuilder.append((String) listElement);
      }  else if (listElement instanceof Double) {
        childBuilder.append((Double) listElement);
      } else if (listElement instanceof Float) {
        childBuilder.append((Float) listElement);
      } else if (listElement instanceof Boolean) {
        childBuilder.append((Boolean) listElement);
      } else if (listElement instanceof Long) {
        childBuilder.append((Long) listElement);
      } else if (listElement instanceof Byte) {
        childBuilder.append((Byte) listElement);
      } else if (listElement instanceof Short) {
        childBuilder.append((Short) listElement);
      } else if (listElement instanceof BigDecimal) {
        childBuilder.append((BigDecimal) listElement);
      } else if (listElement instanceof List) {
        childBuilder.append((List) listElement);
      } else if (listElement instanceof StructData) {
        childBuilder.append((StructData) listElement);
      } else {
        throw new IllegalStateException("Unexpected element type: " + listElement.getClass());
      }
    }

    @Deprecated
    public void incrCurrentIndex() {
      currentIndex =  currentIndex + 1;
    }

    public int getCurrentIndex() {
      return currentIndex;
    }

    public int getCurrentByteIndex() {
      return currentByteIndex;
    }

    public final ColumnBuilder append(byte value) {
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      assert type.isBackedByByte();
      assert currentIndex < rows;
      data.setByte(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      currentByteIndex += type.getSizeInBytes();
      return this;
    }

    public final ColumnBuilder append(short value) {
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      assert type.isBackedByShort();
      assert currentIndex < rows;
      data.setShort(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      currentByteIndex += type.getSizeInBytes();
      return this;
    }

    public final ColumnBuilder append(int value) {
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      assert type.isBackedByInt();
      assert currentIndex < rows;
      data.setInt(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      currentByteIndex += type.getSizeInBytes();
      return this;
    }

    public final ColumnBuilder append(long value) {
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      assert type.isBackedByLong();
      assert currentIndex < rows;
      data.setLong(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      currentByteIndex += type.getSizeInBytes();
      return this;
    }

    public final ColumnBuilder append(float value) {
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      assert type.equals(DType.FLOAT32);
      assert currentIndex < rows;
      data.setFloat(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      currentByteIndex += type.getSizeInBytes();
      return this;
    }

    public final ColumnBuilder append(double value) {
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      assert type.equals(DType.FLOAT64);
      assert currentIndex < rows;
      data.setDouble(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      currentByteIndex += type.getSizeInBytes();
      return this;
    }

    public final ColumnBuilder append(boolean value) {
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      assert type.equals(DType.BOOL8);
      assert currentIndex < rows;
      data.setBoolean(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      currentByteIndex += type.getSizeInBytes();
      return this;
    }

    public final ColumnBuilder append(BigDecimal value) {
      growBuffersAndRows(false, currentIndex * type.getSizeInBytes() + type.getSizeInBytes());
      assert currentIndex < rows;
      // Rescale input decimal with UNNECESSARY policy, which accepts no precision loss.
      BigInteger unscaledVal = value.setScale(-type.getScale(), RoundingMode.UNNECESSARY).unscaledValue();
      if (type.typeId == DType.DTypeEnum.DECIMAL32) {
        data.setInt(currentIndex * type.getSizeInBytes(), unscaledVal.intValueExact());
      } else if (type.typeId == DType.DTypeEnum.DECIMAL64) {
        data.setLong(currentIndex * type.getSizeInBytes(), unscaledVal.longValueExact());
      } else {
        throw new IllegalStateException(type + " is not a supported decimal type.");
      }
      currentIndex++;
      currentByteIndex += type.getSizeInBytes();
      return this;
    }

    public ColumnBuilder append(String value) {
      assert value != null : "appendNull must be used to append null strings";
      return appendUTF8String(value.getBytes(StandardCharsets.UTF_8));
    }

    public ColumnBuilder appendUTF8String(byte[] value) {
      return appendUTF8String(value, 0, value.length);
    }

    public ColumnBuilder appendUTF8String(byte[] value, int srcOffset, int length) {
      assert value != null : "appendNull must be used to append null strings";
      assert srcOffset >= 0;
      assert length >= 0;
      assert value.length + srcOffset <= length;
      assert type.equals(DType.STRING) : " type " + type + " is not String";
      currentIndex++;
      growBuffersAndRows(false, length);
      assert currentIndex < rows + 1;
      if (length > 0) {
        data.setBytes(currentByteIndex, value, srcOffset, length);
      }
      currentByteIndex += length;
      offsets.setInt(currentIndex * OFFSET_SIZE, currentByteIndex);
      return this;
    }

    public ColumnBuilder getChild(int index) {
      return childBuilders.get(index);
    }

    /**
     * Finish and create the immutable ColumnVector, copied to the device.
     */
    public final ColumnVector buildAndPutOnDevice() {
      try (HostColumnVector tmp = build()) {
        return tmp.copyToDevice();
      }
    }

    @Override
    public void close() {
      if (!built) {
        if (data != null) {
          data.close();
          data = null;
        }
        if (valid != null) {
          valid.close();
          valid = null;
        }
        if (offsets != null) {
          offsets.close();
          offsets = null;
        }
        for (ColumnBuilder childBuilder : childBuilders) {
          childBuilder.close();
        }
        built = true;
      }
    }

    @Override
    public String toString() {
      StringJoiner sj = new StringJoiner(",");
      for (ColumnBuilder cb : childBuilders) {
        sj.add(cb.toString());
      }
      return "ColumnBuilder{" +
          "type=" + type +
          ", children=" + sj.toString() +
          ", data=" + data +
          ", valid=" + valid +
          ", currentIndex=" + currentIndex +
          ", nullCount=" + nullCount +
          ", estimatedRows=" + estimatedRows +
          ", populatedRows=" + rows +
          ", built=" + built +
          '}';
    }
  }

  public static final class Builder implements AutoCloseable {
    private final long rows;
    private final DType type;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private HostMemoryBuffer offsets;
    private long currentIndex = 0;
    private long nullCount;
    private int currentStringByteIndex = 0;
    private boolean built;

    /**
     * Create a builder with a buffer of size rows
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param stringBufferSize the size of the string data buffer if we are
     *                         working with Strings.  It is ignored otherwise.
     */
    Builder(DType type, long rows, long stringBufferSize) {
      this.type = type;
      this.rows = rows;
      if (type.equals(DType.STRING)) {
        if (stringBufferSize <= 0) {
          // We need at least one byte or we will get NULL back for data
          stringBufferSize = 1;
        }
        this.data = HostMemoryBuffer.allocate(stringBufferSize);
        // The offsets are ints and there is 1 more than the number of rows.
        this.offsets = HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE);
        // The first offset is always 0
        this.offsets.setInt(0, 0);
      } else {
        this.data = HostMemoryBuffer.allocate(rows * type.getSizeInBytes());
      }
    }

    /**
     * Create a builder with a buffer of size rows (for testing ONLY).
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param testData   a buffer to hold the data (should be large enough to hold rows entries).
     * @param testValid  a buffer to hold the validity vector (should be large enough to hold
     *                   rows entries or is null).
     * @param testOffsets a buffer to hold the offsets for strings and string categories.
     */
    Builder(DType type, long rows, HostMemoryBuffer testData,
            HostMemoryBuffer testValid, HostMemoryBuffer testOffsets) {
      this.type = type;
      this.rows = rows;
      this.data = testData;
      this.valid = testValid;
    }

    public final Builder append(boolean value) {
      assert type.equals(DType.BOOL8);
      assert currentIndex < rows;
      data.setByte(currentIndex * type.getSizeInBytes(), value ? (byte)1 : (byte)0);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value) {
      assert type.isBackedByByte();
      assert currentIndex < rows;
      data.setByte(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value, long count) {
      assert (count + currentIndex) <= rows;
      assert type.isBackedByByte();
      data.setMemory(currentIndex * type.getSizeInBytes(), count, value);
      currentIndex += count;
      return this;
    }

    public final Builder append(short value) {
      assert type.isBackedByShort();
      assert currentIndex < rows;
      data.setShort(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(int value) {
      assert type.isBackedByInt();
      assert currentIndex < rows;
      data.setInt(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(long value) {
      assert type.isBackedByLong();
      assert currentIndex < rows;
      data.setLong(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(float value) {
      assert type.equals(DType.FLOAT32);
      assert currentIndex < rows;
      data.setFloat(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(double value) {
      assert type.equals(DType.FLOAT64);
      assert currentIndex < rows;
      data.setDouble(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    /**
     * Append java.math.BigDecimal into HostColumnVector with UNNECESSARY RoundingMode.
     * Input decimal should have a larger scale than column vector.Otherwise, an ArithmeticException will be thrown while rescaling.
     * If unscaledValue after rescaling exceeds the max precision of rapids type,
     * an ArithmeticException will be thrown while extracting integral.
     *
     * @param value BigDecimal value to be appended
     */
    public final Builder append(BigDecimal value) {
      return append(value, RoundingMode.UNNECESSARY);
    }

    /**
     * Append java.math.BigDecimal into HostColumnVector with user-defined RoundingMode.
     * Input decimal will be rescaled according to scale of column type and RoundingMode before appended.
     * If unscaledValue after rescaling exceeds the max precision of rapids type, an ArithmeticException will be thrown.
     *
     * @param value        BigDecimal value to be appended
     * @param roundingMode rounding mode determines rescaling behavior
     */
    public final Builder append(BigDecimal value, RoundingMode roundingMode) {
      assert type.isDecimalType();
      assert currentIndex < rows;
      BigInteger unscaledValue = value.setScale(-type.getScale(), roundingMode).unscaledValue();
      if (type.typeId == DType.DTypeEnum.DECIMAL32) {
        assert value.precision() <= DType.DECIMAL32_MAX_PRECISION : "value exceeds maximum precision for DECIMAL32";
        data.setInt(currentIndex * type.getSizeInBytes(), unscaledValue.intValueExact());
      } else if (type.typeId == DType.DTypeEnum.DECIMAL64) {
        assert value.precision() <= DType.DECIMAL64_MAX_PRECISION : "value exceeds maximum precision for DECIMAL64 ";
        data.setLong(currentIndex * type.getSizeInBytes(), unscaledValue.longValueExact());
      } else {
        throw new IllegalStateException(type + " is not a supported decimal type.");
      }
      currentIndex++;
      return this;
    }

    public final Builder appendUnscaledDecimal(int value) {
      assert type.typeId == DType.DTypeEnum.DECIMAL32;
      assert currentIndex < rows;
      data.setInt(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder appendUnscaledDecimal(long value) {
      assert type.typeId == DType.DTypeEnum.DECIMAL64;
      assert currentIndex < rows;
      data.setLong(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public Builder append(String value) {
      assert value != null : "appendNull must be used to append null strings";
      return appendUTF8String(value.getBytes(StandardCharsets.UTF_8));
    }

    public Builder appendUTF8String(byte[] value) {
      return appendUTF8String(value, 0, value.length);
    }

    public Builder appendUTF8String(byte[] value, int offset, int length) {
      assert value != null : "appendNull must be used to append null strings";
      assert offset >= 0;
      assert length >= 0;
      assert value.length + offset <= length;
      assert type.equals(DType.STRING);
      assert currentIndex < rows;
      // just for strings we want to throw a real exception if we would overrun the buffer
      long oldLen = data.getLength();
      long newLen = oldLen;
      while (currentStringByteIndex + length > newLen) {
        newLen *= 2;
      }
      if (newLen > Integer.MAX_VALUE) {
        throw new IllegalStateException("A string buffer is not supported over 2GB in size");
      }
      if (newLen != oldLen) {
        // need to grow the size of the buffer.
        HostMemoryBuffer newData = HostMemoryBuffer.allocate(newLen);
        try {
          newData.copyFromHostBuffer(0, data, 0, currentStringByteIndex);
          data.close();
          data = newData;
          newData = null;
        } finally {
          if (newData != null) {
            newData.close();
          }
        }
      }
      if (length > 0) {
        data.setBytes(currentStringByteIndex, value, offset, length);
      }
      currentStringByteIndex += length;
      currentIndex++;
      offsets.setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      return this;
    }

    public Builder appendArray(byte... values) {
      assert (values.length + currentIndex) <= rows;
      assert type.isBackedByByte();
      data.setBytes(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(short... values) {
      assert type.isBackedByShort();
      assert (values.length + currentIndex) <= rows;
      data.setShorts(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(int... values) {
      assert type.isBackedByInt();
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(long... values) {
      assert type.isBackedByLong();
      assert (values.length + currentIndex) <= rows;
      data.setLongs(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(float... values) {
      assert type.equals(DType.FLOAT32);
      assert (values.length + currentIndex) <= rows;
      data.setFloats(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(double... values) {
      assert type.equals(DType.FLOAT64);
      assert (values.length + currentIndex) <= rows;
      data.setDoubles(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendUnscaledDecimalArray(int... values) {
      assert type.typeId == DType.DTypeEnum.DECIMAL32;
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendUnscaledDecimalArray(long... values) {
      assert type.typeId == DType.DTypeEnum.DECIMAL64;
      assert (values.length + currentIndex) <= rows;
      data.setLongs(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public Builder appendBoxed(BigDecimal... values) throws IndexOutOfBoundsException {
      assert type.isDecimalType();
      for (BigDecimal v : values) {
        if (v == null) {
          appendNull();
        } else {
          append(v);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Byte... values) throws IndexOutOfBoundsException {
      for (Byte b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Boolean... values) throws IndexOutOfBoundsException {
      for (Boolean b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b ? (byte) 1 : (byte) 0);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Short... values) throws IndexOutOfBoundsException {
      for (Short b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Integer... values) throws IndexOutOfBoundsException {
      for (Integer b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Long... values) throws IndexOutOfBoundsException {
      for (Long b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Float... values) throws IndexOutOfBoundsException {
      for (Float b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Double... values) throws IndexOutOfBoundsException {
      for (Double b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(String... values) throws IndexOutOfBoundsException {
      for (String b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    // TODO see if we can remove this...
    /**
     * Append this vector to the end of this vector
     * @param columnVector - Vector to be added
     * @return - The CudfColumn based on this builder values
     */
    public final Builder append(HostColumnVector columnVector) {
      assert columnVector.rows <= (rows - currentIndex);
      assert columnVector.type.equals(type);

      if (type.equals(DType.STRING)) {
        throw new UnsupportedOperationException(
            "Appending a string column vector client side is not currently supported");
      } else {
        data.copyFromHostBuffer(currentIndex * type.getSizeInBytes(), columnVector.offHeap.data,
            0L,
            columnVector.getRowCount() * type.getSizeInBytes());
      }

      //As this is doing the append on the host assume that a null count is available
      long otherNc = columnVector.getNullCount();
      if (otherNc != 0) {
        if (valid == null) {
          allocateBitmaskAndSetDefaultValues();
        }
        //copy values from intCudfColumn to this
        BitVectorHelper.append(columnVector.offHeap.valid, valid, currentIndex,
            columnVector.rows);
        nullCount += otherNc;
      }
      currentIndex += columnVector.rows;
      return this;
    }

    private void allocateBitmaskAndSetDefaultValues() {
      long bitmaskSize = ColumnView.getNativeValidPointerSize((int) rows);
      valid = HostMemoryBuffer.allocate(bitmaskSize);
      valid.setMemory(0, bitmaskSize, (byte) 0xFF);
    }

    /**
     * Append null value.
     */
    public final Builder appendNull() {
      setNullAt(currentIndex);
      currentIndex++;
      if (type.equals(DType.STRING)) {
        offsets.setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      }
      return this;
    }

    /**
     * Set a specific index to null.
     * @param index
     */
    public final Builder setNullAt(long index) {
      assert index < rows;

      // add null
      if (this.valid == null) {
        allocateBitmaskAndSetDefaultValues();
      }
      nullCount += BitVectorHelper.setNullAt(valid, index);
      return this;
    }

    /**
     * Finish and create the immutable CudfColumn.
     */
    public final HostColumnVector build() {
      HostColumnVector cv = new HostColumnVector(type,
          currentIndex, Optional.of(nullCount), data, valid, offsets);
      built = true;
      return cv;
    }

    /**
     * Finish and create the immutable ColumnVector, copied to the device.
     */
    public final ColumnVector buildAndPutOnDevice() {
      try (HostColumnVector tmp = build()) {
        return tmp.copyToDevice();
      }
    }

    /**
     * Close this builder and free memory if the CudfColumn wasn't generated. Verifies that
     * the data was released even in the case of an error.
     */
    @Override
    public final void close() {
      if (!built) {
        data.close();
        data = null;
        if (valid != null) {
          valid.close();
          valid = null;
        }
        if (offsets != null) {
          offsets.close();
          offsets = null;
        }
        built = true;
      }
    }

    @Override
    public String toString() {
      return "Builder{" +
          "data=" + data +
          "type=" + type +
          ", valid=" + valid +
          ", currentIndex=" + currentIndex +
          ", nullCount=" + nullCount +
          ", rows=" + rows +
          ", built=" + built +
          '}';
    }
  }

  public static abstract class DataType {
    abstract DType getType();
    abstract boolean isNullable();
    abstract DataType getChild(int index);
    abstract int getNumChildren();
  }

  public static class ListType extends HostColumnVector.DataType {
    private boolean isNullable;
    private HostColumnVector.DataType child;

    public ListType(boolean isNullable, DataType child) {
      this.isNullable = isNullable;
      this.child = child;
    }

    @Override
    DType getType() {
      return DType.LIST;
    }

    @Override
    boolean isNullable() {
      return isNullable;
    }

    @Override
    HostColumnVector.DataType getChild(int index) {
      if (index > 0) {
        return null;
      }
      return child;
    }

    @Override
    int getNumChildren() {
      return 1;
    }
  }

  public static class StructData {
    List<Object> dataRecord;

    public StructData(List<Object> dataRecord) {
      this.dataRecord = dataRecord;
    }

    public StructData(Object... data) {
      this(Arrays.asList(data));
    }

    public int getNumFields() {
      if (dataRecord != null) {
        return dataRecord.size();
      } else {
        return 0;
      }
    }
  }

  public static class StructType extends HostColumnVector.DataType {
    private boolean isNullable;
    private List<HostColumnVector.DataType> children;

    public StructType(boolean isNullable, List<HostColumnVector.DataType> children) {
      this.isNullable = isNullable;
      this.children = children;
    }

    public StructType(boolean isNullable, DataType... children) {
      this(isNullable, Arrays.asList(children));
    }

    @Override
    DType getType() {
      return DType.STRUCT;
    }

    @Override
    boolean isNullable() {
      return isNullable;
    }

    @Override
    HostColumnVector.DataType getChild(int index) {
      return children.get(index);
    }

    @Override
    int getNumChildren() {
      return children.size();
    }
  }

  public static class BasicType extends HostColumnVector.DataType {
    private DType type;
    private boolean isNullable;

    public BasicType(boolean isNullable, DType type) {
      this.isNullable = isNullable;
      this.type = type;
    }

    @Override
    DType getType() {
      return type;
    }

    @Override
    boolean isNullable() {
      return isNullable;
    }

    @Override
    HostColumnVector.DataType getChild(int index) {
      return null;
    }

    @Override
    int getNumChildren() {
      return 0;
    }
  }
}
