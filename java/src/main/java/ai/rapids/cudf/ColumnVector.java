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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * A Column Vector. This class represents the immutable vector of data.  This class holds
 * references to off heap memory and is reference counted to know when to release it.  Call
 * close to decrement the reference count when you are done with the column, and call inRefCount
 * to increment the reference count.
 */
public final class ColumnVector implements AutoCloseable, BinaryOperable {
  static final boolean REF_COUNT_DEBUG = Boolean.getBoolean("ai.rapids.refcount.debug");
  private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final DType type;
  private final OffHeapState offHeap = new OffHeapState();
  // Time Unit of a TIMESTAMP vector
  private TimeUnit tsTimeUnit;
  private long rows;
  private long nullCount;
  private int refCount;

  /**
   * Convert elements in it to a String and join them together. Only use for debug messages
   * where the code execution itself can be disabled as this is not fast.
   */
  private static <T> String stringJoin(String delim, Iterable<T> it) {
    return String.join(delim,
        StreamSupport.stream(it.spliterator(), false)
            .map((i) -> i.toString())
            .collect(Collectors.toList()));
  }

  /**
   * Wrap an existing on device gdf_column with the corresponding ColumnVector.
   */
  ColumnVector(long nativePointer) {
    assert nativePointer != 0;
    ColumnVectorCleaner.register(this, offHeap);
    offHeap.nativeCudfColumnHandle = nativePointer;
    this.type = getDType(nativePointer);
    offHeap.hostData = null;
    this.rows = getRowCount(nativePointer);
    this.nullCount = getNullCount(nativePointer);
    this.tsTimeUnit = getTimeUnit(nativePointer);
    DeviceMemoryBuffer data = null;
    // The data pointer for a STRING is a pointer to an NVStrings object
    // it will be released when the gdf_column is released. We do not
    // keep a reference to it in the java code.
    if (type != DType.STRING) {
      data = new DeviceMemoryBuffer(getDataPtr(nativePointer), this.rows * type.sizeInBytes);
    }
    DeviceMemoryBuffer valid = null;
    long validPtr = getValidPtr(nativePointer);
    if (validPtr != 0) {
      // We are not using the BitVectorHelper.getValidityAllocationSizeInBytes() because
      // cudfColumn was initialized by cudf and not by cudfjni
      valid = new DeviceMemoryBuffer(validPtr, BitVectorHelper.getValidityLengthInBytes(rows));
    }
    this.offHeap.deviceData = new BufferEncapsulator<>(data, valid, null);
    this.refCount = 0;
    incRefCount();
  }

  /**
   * Create a new column vector with data populated on the host.
   */
  ColumnVector(DType type, TimeUnit tsTimeUnit, long rows, long nullCount,
               HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer) {
    this(type, tsTimeUnit, rows, nullCount, hostDataBuffer, hostValidityBuffer, null);
  }

  /**
   * Create a new column vector with data populated on the host.
   * @param type               the type of the vector
   * @param tsTimeUnit         the time unit, only really applicable for DType.TIMESTAMP
   * @param rows               the number of rows in the vector.
   * @param nullCount          the number of nulls in the vector.
   * @param hostDataBuffer     The host side data for the vector. In the case of STRING and
   *                           STRING_CATEGORY this is the string data stored as bytes.
   * @param hostValidityBuffer arrow like validity buffer 1 bit per row, with padding for
   *                           64-bit alignment.
   * @param offsetBuffer       only valid for STRING and STRING_CATEGORY this is the offsets into
   *                           the hostDataBuffer indicating the start and end of a string
   *                           entry. It
   *                           should be (rows + 1) ints.
   */
  ColumnVector(DType type, TimeUnit tsTimeUnit, long rows, long nullCount,
               HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
               HostMemoryBuffer offsetBuffer) {
    if (nullCount > 0 && hostValidityBuffer == null) {
      throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
    }
    if (type == DType.STRING_CATEGORY || type == DType.STRING) {
      assert offsetBuffer != null : "offsets must be provided for STRING and STRING_CATEGORY";
    } else {
      assert offsetBuffer == null : "offsets are only supported for STRING and STRING_CATEGORY";
    }
    if (type == DType.TIMESTAMP) {
      if (tsTimeUnit == TimeUnit.NONE) {
        this.tsTimeUnit = TimeUnit.MILLISECONDS;
      } else {
        this.tsTimeUnit = tsTimeUnit;
      }
    } else {
      this.tsTimeUnit = TimeUnit.NONE;
    }
    ColumnVectorCleaner.register(this, offHeap);
    offHeap.hostData = new BufferEncapsulator(hostDataBuffer, hostValidityBuffer, offsetBuffer);
    offHeap.deviceData = null;
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;
    refCount = 0;
    incRefCount();
  }

  /**
   * Create a new column vector for specific tests that track buffers already on the device.
   */
  ColumnVector(DType type, TimeUnit tsTimeUnit, long rows,
               DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer) {
    assert type != DType.STRING && type != DType.STRING_CATEGORY : "STRING AND STRING_CATEGORY " +
        "NOT SUPPORTED BY THIS CONSTRUCTOR";
    ColumnVectorCleaner.register(this, offHeap);
    if (type == DType.TIMESTAMP) {
      if (tsTimeUnit == TimeUnit.NONE) {
        this.tsTimeUnit = TimeUnit.MILLISECONDS;
      } else {
        this.tsTimeUnit = tsTimeUnit;
      }
    } else {
      this.tsTimeUnit = TimeUnit.NONE;
    }
    offHeap.deviceData = new BufferEncapsulator(dataBuffer, validityBuffer, null);
    offHeap.hostData = null;
    this.rows = rows;
    // This should be overwritten, as this constructor is just for output
    this.nullCount = 0;
    this.type = type;
    refCount = 0;
    incRefCount();
  }

  /**
   * Close this Vector and free memory allocated for HostMemoryBuffer and DeviceMemoryBuffer
   */
  @Override
  public final void close() {
    refCount--;
    offHeap.delRef();
    if (refCount == 0) {
      offHeap.clean(false);
    } else if (refCount < 0) {
      log.error("Close called too many times on {}", this);
      offHeap.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times");
    }
  }

  @Override
  public String toString() {
    return "ColumnVector{" +
        "rows=" + rows +
        ", type=" + type +
        ", hostData=" + offHeap.hostData +
        ", deviceData=" + offHeap.deviceData +
        ", nullCount=" + nullCount +
        ", cudfColumn=" + offHeap.nativeCudfColumnHandle +
        '}';
  }

  /////////////////////////////////////////////////////////////////////////////
  // METADATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Increment the reference count for this column.  You need to call close on this
   * to decrement the reference count again.
   */
  public ColumnVector incRefCount() {
    refCount++;
    offHeap.addRef();
    return this;
  }

  /**
   * Returns the number of rows in this vector.
   */
  public long getRowCount() {
    return rows;
  }

  /**
   * Returns the type of this vector.
   */
  @Override
  public DType getType() {
    return type;
  }

  /**
   * Returns the number of nulls in the data.
   */
  public long getNullCount() {
    return nullCount;
  }

  /**
   * Returns if the vector has a validity vector allocated or not.
   */
  public boolean hasValidityVector() {
    boolean ret;
    if (offHeap.hostData != null) {
      ret = (offHeap.hostData.valid != null);
    } else {
      ret = (offHeap.deviceData.valid != null);
    }
    return ret;
  }

  /**
   * Returns if the vector has nulls.
   */
  public boolean hasNulls() {
    return getNullCount() > 0;
  }

  /**
   * For vector types that support a TimeUnit (TIMESTAMP),
   * get the unit of time. Will be NONE for vectors that
   * did not have one set.  For a TIMESTAMP NONE is the default
   * unit which should be the same as MILLISECONDS.
   */
  public TimeUnit getTimeUnit() {
    return tsTimeUnit;
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATA MOVEMENT
  /////////////////////////////////////////////////////////////////////////////

  private void checkHasDeviceData() {
    if (offHeap.deviceData == null && rows != 0) {
      throw new IllegalStateException("Vector not on Device");
    }
  }

  private void checkHasHostData() {
    if (offHeap.hostData == null && rows != 0) {
      throw new IllegalStateException("Vector not on Host");
    }
  }

  /**
   * Be sure the data is on the device.
   */
  public final void ensureOnDevice() {
    if (offHeap.deviceData == null && rows != 0) {
      checkHasHostData();

      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
        assert (offHeap.hostData.offsets != null);
      }
      DeviceMemoryBuffer deviceDataBuffer = null;
      DeviceMemoryBuffer deviceValidityBuffer = null;

      // for type == DType.STRING the data buffer in the string is an instance of NVStrings
      // and is allocated/populated later in the call to cudfColumnViewStrings
      if (type == DType.STRING_CATEGORY) {
        // The data buffer holds the indexes into the strings dictionary which is
        // allocated in the call to cudfColumnViewStrings.
        deviceDataBuffer = DeviceMemoryBuffer.allocate(rows * type.sizeInBytes);
      } else if (type != DType.STRING) {
        deviceDataBuffer = DeviceMemoryBuffer.allocate(offHeap.hostData.data.getLength());
      }

      boolean needsCleanup = true;
      try {
        if (hasNulls()) {
          deviceValidityBuffer = DeviceMemoryBuffer.allocate(offHeap.hostData.valid.getLength());
        }
        offHeap.deviceData = new BufferEncapsulator(deviceDataBuffer, deviceValidityBuffer, null);
        needsCleanup = false;
      } finally {
        if (needsCleanup) {
          if (deviceDataBuffer != null) {
            deviceDataBuffer.close();
          }
          if (deviceValidityBuffer != null) {
            deviceValidityBuffer.close();
          }
        }
      }

      if (offHeap.deviceData.valid != null) {
        offHeap.deviceData.valid.copyFromHostBuffer(offHeap.hostData.valid);
      }

      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
        // In the case of STRING and STRING_CATEGORY the gdf_column holds references
        // to the device data that the java code does not, so we will not be lazy about
        // creating the gdf_column instance.
        offHeap.nativeCudfColumnHandle = allocateCudfColumn();
        cudfColumnViewStrings(offHeap.nativeCudfColumnHandle,
            offHeap.hostData.data.getAddress(),
            offHeap.hostData.offsets.getAddress(),
            offHeap.hostData.valid == null ? 0 : offHeap.deviceData.valid.getAddress(),
            offHeap.deviceData.data == null ? 0 : offHeap.deviceData.data.getAddress(),
            (int) rows, type.nativeId,
            (int) getNullCount());
      } else {
        offHeap.deviceData.data.copyFromHostBuffer(offHeap.hostData.data);
      }
    }
  }

  /**
   * Be sure the data is on the host.
   */
  public final void ensureOnHost() {
    if (offHeap.hostData == null && rows != 0) {
      checkHasDeviceData();

      HostMemoryBuffer hostDataBuffer = null;
      HostMemoryBuffer hostValidityBuffer = null;
      HostMemoryBuffer hostOffsetsBuffer = null;
      boolean needsCleanup = true;
      try {
        if (offHeap.deviceData.valid != null) {
          hostValidityBuffer = HostMemoryBuffer.allocate(offHeap.deviceData.valid.getLength());
        }
        if (type == DType.STRING || type == DType.STRING_CATEGORY) {
          long[] vals = getStringDataAndOffsetsBack(getNativeCudfColumnAddress());
          hostDataBuffer = new HostMemoryBuffer(vals[0], vals[1]);
          hostOffsetsBuffer = new HostMemoryBuffer(vals[2], vals[3]);
        } else {
          hostDataBuffer = HostMemoryBuffer.allocate(offHeap.deviceData.data.getLength());
        }

        offHeap.hostData = new BufferEncapsulator(hostDataBuffer, hostValidityBuffer,
            hostOffsetsBuffer);
        needsCleanup = false;
      } finally {
        if (needsCleanup) {
          if (hostDataBuffer != null) {
            hostDataBuffer.close();
          }
          if (hostValidityBuffer != null) {
            hostValidityBuffer.close();
          }
        }
      }
      if (type != DType.STRING && type != DType.STRING_CATEGORY) {
        offHeap.hostData.data.copyFromDeviceBuffer(offHeap.deviceData.data);
      }
      if (offHeap.hostData.valid != null) {
        offHeap.hostData.valid.copyFromDeviceBuffer(offHeap.deviceData.valid);
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Check if the value at index is null or not.
   */
  public boolean isNull(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    if (hasNulls()) {
      checkHasHostData();
      return BitVectorHelper.isNull(offHeap.hostData.valid, index);
    }
    return false;
  }

  /**
   * For testing only.  Allows null checks to go past the number of rows, but not past the end
   * of the buffer.  NOTE: If the validity vector was allocated by cudf itself it is not
   * guaranteed to have the same padding, but for all practical purposes it does.  This is
   * just to verify that the buffer was allocated and initialized properly.
   */
  boolean isNullExtendedRange(long index) {
    long maxNullRow = BitVectorHelper.getValidityAllocationSizeInBytes(rows) * 8;
    assert (index >= 0 && index < maxNullRow) : "TEST: index is out of range 0 <= " + index + " <" +
        " " + maxNullRow;
    if (hasNulls()) {
      checkHasHostData();
      return BitVectorHelper.isNull(offHeap.hostData.valid, index);
    }
    return false;
  }

  /**
   * Generic type independent asserts when getting a value from a single index.
   * @param index where to get the data from.
   */
  private void assertsForGet(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert offHeap.hostData != null : "data is not on the host";
    assert !isNull(index) : " value at " + index + " is null";
  }

  /**
   * Get the value at index.
   */
  public byte getByte(long index) {
    assert type == DType.INT8 || type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.hostData.data.getByte(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final short getShort(long index) {
    assert type == DType.INT16;
    assertsForGet(index);
    return offHeap.hostData.data.getShort(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final int getInt(long index) {
    assert type == DType.INT32 || type == DType.DATE32;
    assertsForGet(index);
    return offHeap.hostData.data.getInt(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final long getLong(long index) {
    assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
    assertsForGet(index);
    return offHeap.hostData.data.getLong(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final float getFloat(long index) {
    assert type == DType.FLOAT32;
    assertsForGet(index);
    return offHeap.hostData.data.getFloat(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final double getDouble(long index) {
    assert type == DType.FLOAT64;
    assertsForGet(index);
    return offHeap.hostData.data.getDouble(index * type.sizeInBytes);
  }

  /**
   * Get the boolean value at index
   */
  public final boolean getBoolean(long index) {
    assert type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.hostData.data.getBoolean(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.  This API is slow as it has to translate the
   * string representation.  Please use it with caution.
   */
  public String getJavaString(long index) {
    assert type == DType.STRING || type == DType.STRING_CATEGORY;
    assertsForGet(index);
    int start = offHeap.hostData.offsets.getInt(index * 4); // size of an int
    int size = offHeap.hostData.offsets.getInt((index + 1) * 4) - start;
    byte[] rawData = new byte[size];
    offHeap.hostData.data.getBytes(rawData, 0, start, size);
    return new String(rawData, StandardCharsets.UTF_8);
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATE/TIME
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Get year from DATE32, DATE64, or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector year() {
    assert type == DType.DATE32 || type == DType.DATE64 || type == DType.TIMESTAMP;
    return new ColumnVector(Cudf.gdfExtractDatetimeYear(this));
  }

  /**
   * Get month from DATE32, DATE64, or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector month() {
    assert type == DType.DATE32 || type == DType.DATE64 || type == DType.TIMESTAMP;
    return new ColumnVector(Cudf.gdfExtractDatetimeMonth(this));
  }

  /**
   * Get day from DATE32, DATE64, or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector day() {
    assert type == DType.DATE32 || type == DType.DATE64 || type == DType.TIMESTAMP;
    return new ColumnVector(Cudf.gdfExtractDatetimeDay(this));
  }

  /**
   * Get hour from DATE64 or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector hour() {
    assert type == DType.DATE64 || type == DType.TIMESTAMP;
    return new ColumnVector(Cudf.gdfExtractDatetimeHour(this));
  }

  /**
   * Get minute from DATE64 or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector minute() {
    assert type == DType.DATE64 || type == DType.TIMESTAMP;
    return new ColumnVector(Cudf.gdfExtractDatetimeMinute(this));
  }

  /**
   * Get second from DATE64 or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector second() {
    assert type == DType.DATE64 || type == DType.TIMESTAMP;
    return new ColumnVector(Cudf.gdfExtractDatetimeSecond(this));
  }

  /////////////////////////////////////////////////////////////////////////////
  // ARITHMETIC
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Multiple different unary operations. The output is the same type as input.
   * @param op      the operation to perform
   * @return the result
   */
  public ColumnVector unaryOp(UnaryOp op) {
    return new ColumnVector(Cudf.gdfUnaryMath(this, op, type));
  }

  /**
   * Calculate the sin, output is the same type as input.
   */
  public ColumnVector sin() {
    return unaryOp(UnaryOp.SIN);
  }

  /**
   * Calculate the cos, output is the same type as input.
   */
  public ColumnVector cos() {
    return unaryOp(UnaryOp.COS);
  }

  /**
   * Calculate the tan, output is the same type as input.
   */
  public ColumnVector tan() {
    return unaryOp(UnaryOp.TAN);
  }

  /**
   * Calculate the arcsin, output is the same type as input.
   */
  public ColumnVector arcsin() {
    return unaryOp(UnaryOp.ARCSIN);
  }

  /**
   * Calculate the arccos, output is the same type as input.
   */
  public ColumnVector arccos() {
    return unaryOp(UnaryOp.ARCCOS);
  }

  /**
   * Calculate the arctan, output is the same type as input.
   */
  public ColumnVector arctan() {
    return unaryOp(UnaryOp.ARCTAN);
  }

  /**
   * Calculate the exp, output is the same type as input.
   */
  public ColumnVector exp() {
    return unaryOp(UnaryOp.EXP);
  }

  /**
   * Calculate the log, output is the same type as input.
   */
  public ColumnVector log() {
    return unaryOp(UnaryOp.LOG);
  }

  /**
   * Calculate the sqrt, output is the same type as input.
   */
  public ColumnVector sqrt() {
    return unaryOp(UnaryOp.SQRT);
  }

  /**
   * Calculate the ceil, output is the same type as input.
   */
  public ColumnVector ceil() {
    return unaryOp(UnaryOp.CEIL);
  }

  /**
   * Calculate the floor, output is the same type as input.
   */
  public ColumnVector floor() {
    return unaryOp(UnaryOp.FLOOR);
  }

  /**
   * Calculate the abs, output is the same type as input.
   */
  public ColumnVector abs() {
    return unaryOp(UnaryOp.ABS);
  }

  /**
   * invert the bits, output is the same type as input.
   */
  public ColumnVector bitInvert() {
    return unaryOp(UnaryOp.BIT_INVERT);
  }

  /**
   * Multiple different binary operations.
   * @param op      the operation to perform
   * @param rhs     the rhs of the operation
   * @param outType the type of output you want.
   * @return the result
   */
  @Override
  public ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
    if (rhs instanceof ColumnVector) {
      ColumnVector cvRhs = (ColumnVector) rhs;
      assert rows == cvRhs.getRowCount();
      return new ColumnVector(Cudf.gdfBinaryOp(this, cvRhs, op, outType));
    } else if (rhs instanceof Scalar) {
      Scalar sRhs = (Scalar) rhs;
      return new ColumnVector(Cudf.gdfBinaryOp(this, sRhs, op, outType));
    } else {
      throw new IllegalArgumentException(rhs.getClass() + " is not supported as a binary op" +
          " with ColumnVector");
    }
  }

  /**
   * Filters a column using a column of boolean values as a mask.
   * <p>
   * Given an input column and a mask column, an element `i` from the input column
   * is copied to the output if the corresponding element `i` in the mask is
   * non-null and `true`. This operation is stable: the input order is preserved.
   * <p>
   * The input and mask columns must be of equal size.
   * <p>
   * The output column has size equal to the number of elements in boolean_mask
   * that are both non-null and `true`.
   * <p>
   * If the input size is zero, there is no error, and an empty column is returned.
   * @param mask column of type {@link DType#BOOL8} used as a mask to filter
   *             the input column
   * @return column containing copy of all elements of this column passing
   * the filter defined by the boolean mask
   */
  public ColumnVector filter(ColumnVector mask) {
    assert mask.getType() == DType.BOOL8;
    assert rows == 0 || rows == mask.getRowCount();
    return new ColumnVector(Cudf.filter(this, mask));
  }

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar sum() {
    return sum(type);
  }

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar sum(DType outType) {
    return reduction(ReductionOp.SUM, outType);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar min() {
    return min(type);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar min(DType outType) {
    return reduction(ReductionOp.MIN, outType);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar max() {
    return max(type);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar max(DType outType) {
    return reduction(ReductionOp.MAX, outType);
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar product() {
    return product(type);
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar product(DType outType) {
    return reduction(ReductionOp.PRODUCT, outType);
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the same type as this column.
   */
  public Scalar sumOfSquares() {
    return sumOfSquares(type);
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the specified type.
   */
  public Scalar sumOfSquares(DType outType) {
    return reduction(ReductionOp.SUMOFSQUARES, outType);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * The null values are skipped for the operation.
   * @param op The reduction operation to perform
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduction(ReductionOp op) {
    return reduction(op, type);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * supported for reduction of non-arithmetic types (DATE32, TIMESTAMP...)
   * The null values are skipped for the operation.
   * @param op      The reduction operation to perform
   * @param outType The type of scalar value to return
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduction(ReductionOp op, DType outType) {
    return Cudf.reduction(this, op, outType);
  }

  /////////////////////////////////////////////////////////////////////////////
  // LOGICAL
  /////////////////////////////////////////////////////////////////////////////
 
  /**
   * Returns a vector of the logical `not` of each value in the input 
   * column (this)
   */
  public ColumnVector not() {
    return unaryOp(UnaryOp.NOT);
  }

  /////////////////////////////////////////////////////////////////////////////
  // TYPE CAST
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Generic method to cast ColumnVector
   * When casting from a Date, Timestamp, or Boolean to a numerical type the underlying numerical
   * representationof the data will be used for the cast. In the cast of Timestamp this means the
   * TimeUnit is ignored and lost.
   * When casting between Date32, Date64, and Timestamp the units of time are used.
   * @param type type of the resulting ColumnVector
   * @param unit the unit of time, really only applicable for TIMESTAMP.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector castTo(DType type, TimeUnit unit) {
    if (this.type == type && this.tsTimeUnit == unit) {
      // Optimization
      return incRefCount();
    }
    return new ColumnVector(Cudf.gdfCast(this, type, unit));
  }

  /**
   * Cast to Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asBytes() {
    return castTo(DType.INT8, TimeUnit.NONE);
  }

  /**
   * Cast to Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asShorts() {
    return castTo(DType.INT16, TimeUnit.NONE);
  }

  /**
   * Cast to Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asInts() {
    return castTo(DType.INT32, TimeUnit.NONE);
  }

  /**
   * Cast to Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asLongs() {
    return castTo(DType.INT64, TimeUnit.NONE);
  }

  /**
   * Cast to Float - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to float
   * When casting from a Date, Timestamp, or Boolean to a float type the underlying numerical
   * representatio of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asFloats() {
    return castTo(DType.FLOAT32, TimeUnit.NONE);
  }

  /**
   * Cast to Double - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to double
   * When casting from a Date, Timestamp, or Boolean to a double type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asDoubles() {
    return castTo(DType.FLOAT64, TimeUnit.NONE);
  }

  /**
   * Cast to Date32 - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to date32
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asDate32() {
    return castTo(DType.DATE32, TimeUnit.NONE);
  }

  /**
   * Cast to Date64 - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to date64
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asDate64() {
    return castTo(DType.DATE64, TimeUnit.NONE);
  }

  /**
   * Cast to Timestamp - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to timestamp
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestamp(TimeUnit unit) {
    return castTo(DType.TIMESTAMP, unit);
  }

  /**
   * Cast to Strings.
   * @return A new vector allocated on the GPU.
   */
  public ColumnVector asStrings() {
    return castTo(DType.STRING, TimeUnit.NONE);
  }

  /**
   * Cast to String Categories.
   * @return A new vector allocated on the GPU.
   */
  public ColumnVector asStringCategories() {
    return castTo(DType.STRING_CATEGORY, TimeUnit.NONE);
  }

  /////////////////////////////////////////////////////////////////////////////
  // STRING CATEGORY METHODS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Returns the category index of the specified string scalar.
   * @param s a {@link Scalar} of type {@link DType#STRING} to lookup
   * @return an integer {@link Scalar} containing the category index or -1
   * if the string was not found in the category.
   */
  public Scalar getCategoryIndex(Scalar s) {
    if (s.getType() != DType.STRING) {
      throw new IllegalArgumentException("scalar must be a string type");
    }
    return Scalar.fromInt(Cudf.getCategoryIndex(this, s));
  }

  /////////////////////////////////////////////////////////////////////////////
  // INTERNAL/NATIVE ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * USE WITH CAUTION: This method exposes the address of the native cudf_column.  This allows
   * writing custom kernels or other cuda operations on the data.  DO NOT close this column
   * vector until you are completely done using the native column.  DO NOT modify the column in
   * any way.  This should be treated as a read only data structure. This API is unstable as
   * the underlying C/C++ API is still not stabilized.  If the underlying data structure
   * is renamed this API may be replaced.  The underlying data structure can change from release
   * to release (it is not stable yet) so be sure that your native code is complied against the
   * exact same version of libcudf as this is released for.
   */
  public final long getNativeCudfColumnAddress() {
    if (offHeap.nativeCudfColumnHandle == 0) {
      assert rows <= Integer.MAX_VALUE;
      assert getNullCount() <= Integer.MAX_VALUE;
      checkHasDeviceData();
      offHeap.nativeCudfColumnHandle = allocateCudfColumn();
      long dataAddr = 0;
      long validAddr = 0;
      if (rows != 0) {
        dataAddr = offHeap.deviceData.data.getAddress();
        if (offHeap.deviceData.valid != null) {
          validAddr = offHeap.deviceData.valid.getAddress();
        }
      }
      cudfColumnViewAugmented(offHeap.nativeCudfColumnHandle,
          dataAddr, validAddr,
          (int) rows, type.nativeId,
          (int) getNullCount(), tsTimeUnit.getNativeId());
    }
    return offHeap.nativeCudfColumnHandle;
  }

  private static native long allocateCudfColumn() throws CudfException;

  /**
   * Set a CuDF column given data and validity bitmask pointers, size, and datatype, and
   * count of null (non-valid) elements
   * @param cudfColumnHandle native handle of gdf_column.
   * @param dataPtr          Pointer to data.
   * @param valid            Pointer to validity bitmask for the data.
   * @param size             Number of rows in the column.
   * @param dtype            Data type of the column.
   * @param null_count       The number of non-valid elements in the validity bitmask.
   * @param timeUnit         {@link TimeUnit}
   */
  private static native void cudfColumnViewAugmented(long cudfColumnHandle, long dataPtr,
                                                     long valid,
                                                     int size, int dtype, int null_count,
                                                     int timeUnit) throws CudfException;

  /**
   * Translate the host side string representation of strings into the device side representation
   * and populate the cudfColumn with it.
   * @param cudfColumnHandle native handle of gdf_column.
   * @param hostDataPtr      Pointer to string data on the host.
   * @param hostOffsetsPtr   Pointer to offsets data on the host.
   * @param deviceValidPtr   Pointer to validity bitmask on the device.
   * @param deviceDataPtr    Pointer to where the int category data will be stored for
   *                         STRING_CATEGORY.
   *                         Should be 0 for STRING
   * @param size             Number of rows in the column.
   * @param dtype            Data type of the column. In this case must be STRING or
   *                         STRING_CATEGORY
   * @param nullCount        The number of non-valid elements in the validity bitmask.
   */
  private static native void cudfColumnViewStrings(long cudfColumnHandle, long hostDataPtr,
                                                   long hostOffsetsPtr, long deviceValidPtr,
                                                   long deviceDataPtr,
                                                   int size, int dtype, int nullCount);

  /**
   * Copy the string data to the host.  This is a little ugly because the addresses
   * returned were allocated by native code but will be freed through java's Unsafe API.
   * In practice this should work so long as we don't try to replace malloc, and java does not.
   * If this does become a problem we can subclass HostMemoryBuffer and add in another JNI
   * call to free using native code.
   * @param cudfColumnHandle the device side cudf column.
   * @return [data address, data length, offsets address, offsets length]
   */
  private static native long[] getStringDataAndOffsetsBack(long cudfColumnHandle);

  static native void freeCudfColumn(long cudfColumnHandle, boolean isDeepClean) throws CudfException;

  private static native long getDataPtr(long cudfColumnHandle) throws CudfException;

  private static native long getValidPtr(long cudfColumnHandle) throws CudfException;

  private static native int getRowCount(long cudfColumnHandle) throws CudfException;

  private static DType getDType(long cudfColumnHandle) throws CudfException {
    return DType.fromNative(getDTypeInternal(cudfColumnHandle));
  }

  private static native int getDTypeInternal(long cudfColumnHandle) throws CudfException;

  private static TimeUnit getTimeUnit(long cudfColumnHandle) throws CudfException {
    return TimeUnit.fromNative(getTimeUnitInternal(cudfColumnHandle));
  }

  private static native int getTimeUnitInternal(long cudfColumnHandle) throws CudfException;

  private static native int getNullCount(long cudfColumnHandle) throws CudfException;

  private static native long concatenate(long[] columnHandles) throws CudfException;

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Encapsulator class to hold the two buffers as a cohesive object
   */
  private static final class BufferEncapsulator<T extends MemoryBuffer> implements AutoCloseable {
    public final T data;
    public final T valid;
    public final T offsets;

    BufferEncapsulator(T data, T valid, T offsets) {
      this.data = data;
      this.valid = valid;
      this.offsets = offsets;
    }

    @Override
    public String toString() {
      T type = data == null ? valid : data;
      type = type == null ? offsets : type;
      String t = "UNKNOWN";
      if (type != null) {
        t = type.getClass().getSimpleName();
      }
      return "BufferEncapsulator{type= " + t
          + ", data= " + data
          + ", valid= " + valid
          + ", offsets= " + offsets + "}";
    }

    @Override
    public void close() {
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

  /**
   * When debug is enabled holds information about inc and dec of ref count.
   */
  private static final class RefCountDebugItem {
    final StackTraceElement[] stackTrace;
    final long timeMs;
    final String op;

    public RefCountDebugItem(String op) {
      this.stackTrace = Thread.currentThread().getStackTrace();
      this.timeMs = System.currentTimeMillis();
      this.op = op;
    }

    public String toString() {
      Date date = new Date(timeMs);
      // Simple Date Format is horribly expensive only do this when debug is turned on!
      SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSSS z");
      return dateFormat.format(date) + ": " + op + "\n"
          + stringJoin("\n", Arrays.asList(stackTrace))
          + "\n";
    }
  }

  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  protected static final class OffHeapState implements ColumnVectorCleaner.Cleaner {
    private final List<RefCountDebugItem> refCountDebug;
    public BufferEncapsulator<HostMemoryBuffer> hostData;
    public BufferEncapsulator<DeviceMemoryBuffer> deviceData;
    private long nativeCudfColumnHandle = 0;

    public OffHeapState() {
      if (REF_COUNT_DEBUG) {
        refCountDebug = new LinkedList<>();
      } else {
        refCountDebug = null;
      }
    }

    public final void addRef() {
      if (REF_COUNT_DEBUG) {
        refCountDebug.add(new RefCountDebugItem("INC"));
      }
    }

    public final void delRef() {
      if (REF_COUNT_DEBUG) {
        refCountDebug.add(new RefCountDebugItem("DEC"));
      }
    }

    public final void logRefCountDebug(String message) {
      if (REF_COUNT_DEBUG) {
        log.error("{}: {}", message, stringJoin("\n", refCountDebug));
      }
    }

    @Override
    public boolean clean(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (hostData != null) {
        hostData.close();
        hostData = null;
        neededCleanup = true;
      }
      if (deviceData != null) {
        deviceData.close();
        deviceData = null;
        neededCleanup = true;
      }
      if (nativeCudfColumnHandle != 0) {
        freeCudfColumn(nativeCudfColumnHandle, false);
        nativeCudfColumnHandle = 0;
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("YOU LEAKED A COLUMN VECTOR!!!!");
        logRefCountDebug("Leaked vector");
      }
      return neededCleanup;
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
    return new Builder(type, TimeUnit.NONE, rows);
  }

  /**
   * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when
   * done with it. Please try to use {@see #build(int, Consumer)} instead to avoid needing to
   * close the builder.
   * @param type the type of vector to build.
   * @param rows the number of rows this builder can hold
   * @return the builder to use.
   */
  public static Builder builder(DType type, TimeUnit tsTimeUnit, int rows) {
    return new Builder(type, tsTimeUnit, rows);
  }

  /**
   * Create a new vector.
   * @param type the type of vector to build.
   * @param rows maximum number of rows that the vector can hold.
   * @param init what will initialize the vector.
   * @return the created vector.
   */
  public static ColumnVector build(DType type, int rows, Consumer<Builder> init) {
    return build(type, TimeUnit.NONE, rows, init);
  }

  /**
   * Create a new vector.
   * @param type       the type of vector to build.
   * @param tsTimeUnit the unit of time, really only applicable for TIMESTAMP.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static ColumnVector build(DType type, TimeUnit tsTimeUnit, int rows,
                                   Consumer<Builder> init) {
    try (Builder builder = builder(type, tsTimeUnit, rows)) {
      init.accept(builder);
      return builder.build();
    }
  }

  /**
   * Create a new vector without sending data to the device.
   * @param type the type of vector to build.
   * @param rows maximum number of rows that the vector can hold.
   * @param init what will initialize the vector.
   * @return the created vector.
   */
  public static ColumnVector buildOnHost(DType type, int rows, Consumer<Builder> init) {
    return buildOnHost(type, TimeUnit.NONE, rows, init);
  }

  /**
   * Create a new vector without sending data to the device.
   * @param type       the type of vector to build.
   * @param tsTimeUnit the unit of time, really only applicable for TIMESTAMP.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static ColumnVector buildOnHost(DType type, TimeUnit tsTimeUnit, int rows,
                                         Consumer<Builder> init) {
    try (Builder builder = builder(type, tsTimeUnit, rows)) {
      init.accept(builder);
      return builder.buildOnHost();
    }
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector boolFromBytes(byte... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromBytes(byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromShorts(short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromInts(int... values) {
    return build(DType.INT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromLongs(long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromFloats(float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromDoubles(double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector datesFromInts(int... values) {
    return build(DType.DATE32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector datesFromLongs(long... values) {
    return build(DType.DATE64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampsFromLongs(long... values) {
    return build(DType.TIMESTAMP, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampsFromLongs(TimeUnit tsTimeUnit, long... values) {
    return build(DType.TIMESTAMP, tsTimeUnit, values.length, (b) -> b.appendArray(values));
  }

  private static ColumnVector fromStrings(DType type, String... values) {
    HostMemoryBuffer data = null;
    HostMemoryBuffer offsets = null;
    HostMemoryBuffer valid = null;
    ColumnVector ret = null;
    boolean needsCleanup = true;
    try {
      int rows = values.length;
      long nullCount = 0;
      // How many bytes do we need to hold the data.  Sorry this is really expensive
      long bufferSize = 0;
      for (String s : values) {
        if (s == null) {
          nullCount++;
        } else {
          bufferSize += s.getBytes(StandardCharsets.UTF_8).length;
        }
      }
      data = HostMemoryBuffer.allocate(bufferSize);
      if (nullCount > 0) {
        // copy and pasted from allocateBitmaskAndSetDefaultValues
        long bitmaskSize = BitVectorHelper.getValidityAllocationSizeInBytes(rows);
        valid = HostMemoryBuffer.allocate(bitmaskSize);
        valid.setMemory(0, bitmaskSize, (byte) 0xFF);
      }

      offsets = HostMemoryBuffer.allocate((rows + 1) * 4);
      int offset = 0;
      // The initial offset is always 0
      offsets.setInt(0, offset);
      for (int i = 0; i < values.length; i++) {
        String s = values[i];
        if (s == null) {
          BitVectorHelper.setNullAt(valid, i);
        } else {
          byte[] utf8 = s.getBytes(StandardCharsets.UTF_8);
          data.setBytes(offset, utf8, 0, utf8.length);
          offset += utf8.length;
        }
        offsets.setInt((i + 1L) * 4, offset);
      }
      ret = new ColumnVector(type, TimeUnit.NONE, rows, nullCount, data, valid, offsets);
      ret.ensureOnDevice();
      needsCleanup = false;
      return ret;
    } finally {
      if (needsCleanup) {
        if (ret != null) {
          ret.close();
        } else {
          if (data != null) {
            data.close();
          }
          if (offsets != null) {
            offsets.close();
          }
          if (valid != null) {
            valid.close();
          }
        }
      }
    }
  }

  /**
   * Create a new category string vector from the given values.  This API
   * supports inline nulls. This is really intended to be used only for testing as
   * it is slow and memory intensive to translate between java strings and UTF8 strings.
   */
  public static ColumnVector categoryFromStrings(String... values) {
    return fromStrings(DType.STRING_CATEGORY, values);
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls. This is really intended to be used only for testing as
   * it is slow and memory intensive to translate between java strings and UTF8 strings.
   */
  public static ColumnVector fromStrings(String... values) {
    return fromStrings(DType.STRING, values);
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedBooleans(Boolean... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedBytes(Byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedShorts(Short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedInts(Integer... values) {
    return build(DType.INT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedLongs(Long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedFloats(Float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedDoubles(Double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector datesFromBoxedInts(Integer... values) {
    return build(DType.DATE32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector datesFromBoxedLongs(Long... values) {
    return build(DType.DATE64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampsFromBoxedLongs(TimeUnit tsTimeUnit, Long... values) {
    return build(DType.TIMESTAMP, tsTimeUnit, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector by concatenating multiple columns together.
   * Note that all columns must have the same type.
   */
  public static ColumnVector concatenate(ColumnVector... columns) {
    if (columns.length < 2) {
      throw new IllegalArgumentException("Concatenate requires 2 or more columns");
    }
    long[] columnHandles = new long[columns.length];
    for (int i = 0; i < columns.length; ++i) {
      columnHandles[i] = columns[i].getNativeCudfColumnAddress();
    }
    return new ColumnVector(concatenate(columnHandles));
  }

  /**
   * Build
   */
  public static final class Builder implements AutoCloseable {
    private final long rows;
    private final DType type;
    private final TimeUnit tsTimeUnit;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private long currentIndex = 0;
    private long nullCount;
    private boolean built;

    /**
     * Create a builder with a buffer of size rows
     * @param type       datatype
     * @param tsTimeUnit for TIMESTAMP the unit of time it is storing.
     * @param rows       number of rows to allocate.
     */
    Builder(DType type, TimeUnit tsTimeUnit, long rows) {
      this.type = type;
      this.tsTimeUnit = tsTimeUnit;
      this.rows = rows;
      this.data = HostMemoryBuffer.allocate(rows * type.sizeInBytes);
    }

    /**
     * Create a builder with a buffer of size rows (for testing ONLY).
     * @param type       datatype
     * @param tsTimeUnit for TIMESTAMP the unit of time it is storing.
     * @param rows       number of rows to allocate.
     * @param testData   a buffer to hold the data (should be large enough to hold rows entries).
     * @param testValid  a buffer to hold the validity vector (should be large enough to hold
     *                   rows entries or is null).
     */
    Builder(DType type, TimeUnit tsTimeUnit, long rows, HostMemoryBuffer testData,
            HostMemoryBuffer testValid) {
      this.type = type;
      this.tsTimeUnit = tsTimeUnit;
      this.rows = rows;
      this.data = testData;
      this.valid = testValid;
    }

    public final Builder append(boolean value) {
      assert type == DType.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value ? (byte)1 : (byte)0);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value) {
      assert type == DType.INT8 || type == DType.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value, long count) {
      assert (count + currentIndex) <= rows;
      assert type == DType.INT8 || type == DType.BOOL8;
      data.setMemory(currentIndex * type.sizeInBytes, count, value);
      currentIndex += count;
      return this;
    }

    public final Builder append(short value) {
      assert type == DType.INT16;
      assert currentIndex < rows;
      data.setShort(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(int value) {
      assert (type == DType.INT32 || type == DType.DATE32);
      assert currentIndex < rows;
      data.setInt(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(long value) {
      assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
      assert currentIndex < rows;
      data.setLong(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(float value) {
      assert type == DType.FLOAT32;
      assert currentIndex < rows;
      data.setFloat(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(double value) {
      assert type == DType.FLOAT64;
      assert currentIndex < rows;
      data.setDouble(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder appendArray(byte... values) {
      assert (values.length + currentIndex) <= rows;
      assert type == DType.INT8 || type == DType.BOOL8;
      data.setBytes(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public final Builder appendArray(short... values) {
      assert type == DType.INT16;
      assert (values.length + currentIndex) <= rows;
      data.setShorts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public final Builder appendArray(int... values) {
      assert (type == DType.INT32 || type == DType.DATE32);
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public final Builder appendArray(long... values) {
      assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
      assert (values.length + currentIndex) <= rows;
      data.setLongs(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public final Builder appendArray(float... values) {
      assert type == DType.FLOAT32;
      assert (values.length + currentIndex) <= rows;
      data.setFloats(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public final Builder appendArray(double... values) {
      assert type == DType.FLOAT64;
      assert (values.length + currentIndex) <= rows;
      data.setDoubles(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
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
     * Append this vector to the end of this vector
     * @param columnVector - Vector to be added
     * @return - The ColumnVector based on this builder values
     */
    public final Builder append(ColumnVector columnVector) {
      assert columnVector.rows <= (rows - currentIndex);
      assert columnVector.type == type;
      assert columnVector.offHeap.hostData != null;

      data.copyFromHostBuffer(currentIndex * type.sizeInBytes, columnVector.offHeap.hostData.data,
          0L,
          columnVector.getRowCount() * type.sizeInBytes);

      if (columnVector.nullCount != 0) {
        if (valid == null) {
          allocateBitmaskAndSetDefaultValues();
        }
        //copy values from intColumnVector to this
        BitVectorHelper.append(columnVector.offHeap.hostData.valid, valid, currentIndex,
            columnVector.rows);
        nullCount += columnVector.nullCount;
      }
      currentIndex += columnVector.rows;
      return this;
    }

    private void allocateBitmaskAndSetDefaultValues() {
      long bitmaskSize = BitVectorHelper.getValidityAllocationSizeInBytes(rows);
      valid = HostMemoryBuffer.allocate(bitmaskSize);
      valid.setMemory(0, bitmaskSize, (byte) 0xFF);
    }

    /**
     * Append null value.
     */
    public final Builder appendNull() {
      setNullAt(currentIndex);
      currentIndex++;
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
     * Finish and create the immutable ColumnVector.
     */
    public final ColumnVector build() {
      if (built) {
        throw new IllegalStateException("Cannot reuse a builder.");
      }
      ColumnVector cv = new ColumnVector(type, tsTimeUnit,
          currentIndex, nullCount, data, valid);
      try {
        cv.ensureOnDevice();
        built = true;
      } finally {
        if (!built) {
          cv.close();
        }
      }
      return cv;
    }

    /**
     * Finish and create the immutable ColumnVector.
     */
    public final ColumnVector buildOnHost() {
      if (built) {
        throw new IllegalStateException("Cannot reuse a builder.");
      }
      ColumnVector cv = new ColumnVector(type, tsTimeUnit,
          currentIndex, nullCount, data, valid);
      built = true;
      return cv;
    }

    /**
     * Close this builder and free memory if the ColumnVector wasn't generated. Verifies that
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
}
