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

import ai.rapids.cudf.HostColumnVector.Builder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * This class represents the immutable vector of data.  This class holds
 * references to device(GPU) memory and is reference counted to know when to release it.  Call
 * close to decrement the reference count when you are done with the column, and call incRefCount
 * to increment the reference count.
 */
public final class ColumnVector implements AutoCloseable, BinaryOperable {
  private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);
  private static final AtomicLong idGen = new AtomicLong(0);

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final DType type;
  private final OffHeapState offHeap;
  private final long rows;
  private Optional<Long> nullCount = Optional.empty();
  private int refCount;
  private final long internalId = idGen.incrementAndGet();

  /**
   * Wrap an existing on device cudf::column with the corresponding ColumnVector.
   */
  ColumnVector(long nativePointer) {
    assert nativePointer != 0;
    offHeap = new OffHeapState(internalId, nativePointer);
    MemoryCleaner.register(this, offHeap);
    this.type = offHeap.getNativeType();
    this.rows = offHeap.getNativeRowCount();

    this.refCount = 0;
    incRefCountInternal(true);

    MemoryListener.deviceAllocation(getDeviceMemorySize(), internalId);
  }

  /**
   * Create a new column vector based off of data already on the device.
   * @param type the type of the vector
   * @param rows the number of rows in this vector.
   * @param nullCount the number of nulls in the dataset.
   * @param dataBuffer the data stored on the device.  The column vector takes ownership of the
   *                   buffer.  Do not use the buffer after calling this.
   * @param validityBuffer an optional validity buffer. Must be provided if nullCount != 0. The
   *                      column vector takes ownership of the buffer. Do not use the buffer
   *                      after calling this.
   * @param offsetBuffer a host buffer required for strings and string categories. The column
   *                    vector takes ownership of the buffer. Do not use the buffer after calling
   *                    this.
   */
  public ColumnVector(DType type, long rows, Optional<Long> nullCount,
      DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer,
      DeviceMemoryBuffer offsetBuffer) {
    if (type != DType.STRING) {
      assert offsetBuffer == null : "offsets are only supported for STRING";
    }

    offHeap = new OffHeapState(internalId, type, (int) rows, nullCount, dataBuffer, validityBuffer, offsetBuffer);
    MemoryCleaner.register(this, offHeap);
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;

    this.refCount = 0;
    incRefCountInternal(true);

    MemoryListener.deviceAllocation(getDeviceMemorySize(), internalId);
  }

  /**
   * This is a very special constructor that should only ever be called by
   * fromViewWithContiguousAllocation.  It takes a cudf::column_view * instead of a cudf::column *.
   * But to maintain memory ownership properly we need to slice the memory in the view off from
   * a separate buffer that actually owns the memory allocation.
   * @param viewAddress the address of the cudf::column_view
   * @param contiguousBuffer the buffer that this is based off of.
   */
  private ColumnVector(long viewAddress, DeviceMemoryBuffer contiguousBuffer) {
    offHeap = new OffHeapState(internalId, viewAddress, contiguousBuffer);
    MemoryCleaner.register(this, offHeap);
    this.type = offHeap.getNativeType();
    this.rows = offHeap.getNativeRowCount();
    // TODO we may want to ask for the null count anyways...
    this.nullCount = Optional.empty();
    this.refCount = 0;
    incRefCountInternal(true);
    MemoryListener.deviceAllocation(getDeviceMemorySize(), internalId);
  }

  static ColumnVector fromViewWithContiguousAllocation(long columnViewAddress, DeviceMemoryBuffer buffer) {
    return new ColumnVector(columnViewAddress, buffer);
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
        ", nullCount=" + nullCount +
        ", offHeap=" + offHeap +
        '}';
  }

  /////////////////////////////////////////////////////////////////////////////
  // METADATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  static long predictSizeFor(long baseSize, long rows, boolean includeValidity) {
    long total = baseSize * rows;
    if (includeValidity) {
      total += BitVectorHelper.getValidityAllocationSizeInBytes(rows);
    }
    return total;
  }

  long predictSizeFor(DType type) {
    return predictSizeFor(type.sizeInBytes, rows, hasValidityVector());
  }

  private long predictSizeForRowMult(long baseSize, double rowMult) {
    long rowGuess = (long)(rows * rowMult);
    return predictSizeFor(baseSize, rowGuess, hasValidityVector());
  }

  /**
   * Increment the reference count for this column.  You need to call close on this
   * to decrement the reference count again.
   */
  public ColumnVector incRefCount() {
    return incRefCountInternal(false);
  }

  private synchronized ColumnVector incRefCountInternal(boolean isFirstTime) {
    offHeap.addRef();
    if (refCount <= 0 && !isFirstTime) {
      offHeap.logRefCountDebug("INC AFTER CLOSE " + this);
      throw new IllegalStateException("Column is already closed");
    }
    refCount++;
    return this;
  }

  /**
   * Returns the number of rows in this vector.
   */
  public long getRowCount() {
    return rows;
  }

  /**
   * Returns the amount of device memory used.
   */
  public long getDeviceMemorySize() {
    return offHeap != null ? offHeap.getDeviceMemorySize() : 0;
  }

  /**
   * Returns the type of this vector.
   */
  @Override
  public DType getType() {
    return type;
  }

  /**
   * Returns the number of nulls in the data. Note that this might end up
   * being a very expensive operation because if the null count is not
   * known it will be calculated.
   */
  public long getNullCount() {
    if (!nullCount.isPresent()) {
      nullCount = Optional.of(offHeap.getNativeNullCount());
    }
    return nullCount.get();
  }

  /**
   * Returns this column's current refcount
   */
  synchronized int getRefCount() {
    return refCount;
  }

  /**
   * Returns if the vector has a validity vector allocated or not.
   */
  public boolean hasValidityVector() {
    return (offHeap.getValid() != null);
  }

  /**
   * Returns if the vector has nulls.  Note that this might end up
   * being a very expensive operation because if the null count is not
   * known it will be calculated.
   */
  public boolean hasNulls() {
    return getNullCount() > 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATA MOVEMENT
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Copy the data to the host.
   */
  public HostColumnVector copyToHost() {
    try (HostPrediction prediction =
             new HostPrediction(getDeviceMemorySize(), "ensureOnHost");
         NvtxRange toHost = new NvtxRange("ensureOnHost", NvtxColor.BLUE)) {
      HostMemoryBuffer hostDataBuffer = null;
      HostMemoryBuffer hostValidityBuffer = null;
      HostMemoryBuffer hostOffsetsBuffer = null;
      BaseDeviceMemoryBuffer valid = offHeap.getValid();
      BaseDeviceMemoryBuffer offsets = offHeap.getOffsets();
      BaseDeviceMemoryBuffer data = offHeap.getData();
      boolean needsCleanup = true;
      try {
        // We don't have a good way to tell if it is cached on the device or recalculate it on
        // the host for now, so take the hit here.
        getNullCount();

        if (valid != null) {
          hostValidityBuffer = HostMemoryBuffer.allocate(valid.getLength());
          hostValidityBuffer.copyFromDeviceBuffer(valid);
        }
        if (offsets != null) {
          hostOffsetsBuffer = HostMemoryBuffer.allocate(offsets.length);
          hostOffsetsBuffer.copyFromDeviceBuffer(offsets);
        }
        // If a strings column is all null values there is no data buffer allocated
        if (data != null) {
          hostDataBuffer = HostMemoryBuffer.allocate(data.length);
          hostDataBuffer.copyFromDeviceBuffer(data);
        }
        HostColumnVector ret = new HostColumnVector(type, rows, nullCount,
            hostDataBuffer, hostValidityBuffer, hostOffsetsBuffer);
        needsCleanup = false;
        return ret;
      } finally {
        if (needsCleanup) {
          if (hostOffsetsBuffer != null) {
            hostOffsetsBuffer.close();
          }
          if (hostDataBuffer != null) {
            hostDataBuffer.close();
          }
          if (hostValidityBuffer != null) {
            hostValidityBuffer.close();
          }
        }
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // RAW DATA ACCESS
  /////////////////////////////////////////////////////////////////////////////


  /**
   * Get access to the raw device buffer for this column.  This is intended to be used with a lot
   * of caution.  The lifetime of the buffer is tied to the lifetime of the column (Do not close
   * the buffer, as the column will take care of it).  Do not modify the contents of the buffer or
   * it might negatively impact what happens on the column.  The data must be on the device for
   * this to work. Strings and string categories do not currently work because their underlying
   * device layout is currently hidden.
   * @param type the type of buffer to get access to.
   * @return the underlying buffer or null if no buffer is associated with it for this column.
   * Please note that if the column is empty there may be no buffers at all associated with the
   * column.
   */
  public BaseDeviceMemoryBuffer getDeviceBufferFor(BufferType type) {
    BaseDeviceMemoryBuffer srcBuffer;
    switch(type) {
      case VALIDITY:
        srcBuffer = offHeap.getValid();
        break;
      case DATA:
        srcBuffer = offHeap.getData();
        break;
      case OFFSET:
        srcBuffer = offHeap.getOffsets();
        break;
      default:
        throw new IllegalArgumentException(type + " is not a supported buffer type.");

    }
    return srcBuffer;
  }

  /////////////////////////////////////////////////////////////////////////////
  // DEVICE METADATA
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Retrieve the number of characters in each string. Null strings will have value of null.
   *
   * @return ColumnVector holding length of string at index 'i' in the original vector
   */
  public ColumnVector getLengths() {
    assert DType.STRING == type : "length only available for String type";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT32), "getLengths")) {
      return new ColumnVector(lengths(getNativeView()));
    }
  }

  /**
   * Retrieve the number of bytes for each string. Null strings will have value of null.
   *
   * @return ColumnVector, where each element at i = byte count of string at index 'i' in the original vector
   */
  public ColumnVector getByteCount() {
    assert type == DType.STRING : "type has to be a String";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT32), "byteCount")) {
      return new ColumnVector(byteCount(getNativeView()));
    }
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is not null, and FALSE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public ColumnVector isNotNull() {
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8), "isNotNull")) {
      return new ColumnVector(isNotNullNative(getNativeView()));
    }
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * FALSE for any entry that is not null, and TRUE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public ColumnVector isNull() {
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8),"isNull")) {
      return new ColumnVector(isNullNative(getNativeView()));
    }
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is NaN, and FALSE if null or a valid floating point value
   * @return - Boolean vector
   */
  public ColumnVector isNan() {
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8),"isNan")) {
      return new ColumnVector(isNanNative(getNativeView()));
    }
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is null or a valid floating point value, FALSE otherwise
   * @return - Boolean vector
   */
  public ColumnVector isNotNan() {
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8),"isNotNan")) {
      return new ColumnVector(isNotNanNative(getNativeView()));
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Replacement
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Returns a vector with all values "oldValues[i]" replaced with "newValues[i]".
   * Warning:
   *    Currently this function doesn't work for Strings or StringCategories.
   *    NaNs can't be replaced in the original vector but regular values can be replaced with NaNs
   *    Nulls can't be replaced in the original vector but regular values can be replaced with Nulls
   *    Mixing of types isn't allowed, the resulting vector will be the same type as the original.
   *      e.g. You can't replace an integer vector with values from a long vector
   *
   * Usage:
   *    this = {1, 4, 5, 1, 5}
   *    oldValues = {1, 5, 7}
   *    newValues = {2, 6, 9}
   *
   *    result = this.findAndReplaceAll(oldValues, newValues);
   *    result = {2, 4, 6, 2, 6}  (1 and 5 replaced with 2 and 6 but 7 wasn't found so no change)
   *
   * @param oldValues - A vector containing values that should be replaced
   * @param newValues - A vector containing new values
   * @return - A new vector containing the old values replaced with new values
   */
  public ColumnVector findAndReplaceAll(ColumnVector oldValues, ColumnVector newValues) {
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "findAndReplace")) {
      return new ColumnVector(findAndReplaceAll(oldValues.getNativeView(), newValues.getNativeView(), this.getNativeView()));
    }
  }

  /**
   * Returns a ColumnVector with any null values replaced with a scalar.
   * The types of the input ColumnVector and Scalar must match, else an error is thrown.
   *
   * @param scalar - Scalar value to use as replacement
   * @return - ColumnVector with nulls replaced by scalar
   */
  public ColumnVector replaceNulls(Scalar scalar) {
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "replaceNulls")) {
      return new ColumnVector(replaceNulls(getNativeView(), scalar.getScalarHandle()));
    }
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other vectors
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the corresponding row is selected from trueValues
   * otherwise the corresponding row from falseValues is selected.
   * Note that trueValues and falseValues vectors must be the same length as this vector,
   * and trueValues and falseValues must have the same data type.
   * @param trueValues the values to select if a row in this column is true
   * @param falseValues the values to select if a row in this column is not true
   * @return the computed vector
   */
  public ColumnVector ifElse(ColumnVector trueValues, ColumnVector falseValues) {
    if (type != DType.BOOL8) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseVV(getNativeView(), trueValues.getNativeView(), falseValues.getNativeView());
    return new ColumnVector(result);
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other inputs
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the corresponding row is selected from trueValues
   * otherwise the value from falseValue is selected.
   * Note that trueValues must be the same length as this vector,
   * and trueValues and falseValue must have the same data type.
   * Note that the trueValues vector and falseValue scalar must have the same data type.
   * @param trueValues the values to select if a row in this column is true
   * @param falseValue the value to select if a row in this column is not true
   * @return the computed vector
   */
  public ColumnVector ifElse(ColumnVector trueValues, Scalar falseValue) {
    if (type != DType.BOOL8) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseVS(getNativeView(), trueValues.getNativeView(), falseValue.getScalarHandle());
    return new ColumnVector(result);
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other inputs
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the value from trueValue is selected
   * otherwise the corresponding row from falseValues is selected.
   * Note that falseValues must be the same length as this vector,
   * and trueValue and falseValues must have the same data type.
   * Note that the trueValue scalar and falseValues vector must have the same data type.
   * @param trueValue the value to select if a row in this column is true
   * @param falseValues the values to select if a row in this column is not true
   * @return the computed vector
   */
  public ColumnVector ifElse(Scalar trueValue, ColumnVector falseValues) {
    if (type != DType.BOOL8) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseSV(getNativeView(), trueValue.getScalarHandle(), falseValues.getNativeView());
    return new ColumnVector(result);
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other inputs
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the value from trueValue is selected
   * otherwise the value from falseValue is selected.
   * Note that the trueValue and falseValue scalars must have the same data type.
   * @param trueValue the value to select if a row in this column is true
   * @param falseValue the value to select if a row in this column is not true
   * @return the computed vector
   */
  public ColumnVector ifElse(Scalar trueValue, Scalar falseValue) {
    if (type != DType.BOOL8) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseSS(getNativeView(), trueValue.getScalarHandle(), falseValue.getScalarHandle());
    return new ColumnVector(result);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Slice/Split and Concatenate
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Slices a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them
   *
   * The "slice" function divides part of the input column into multiple intervals
   * of rows using the indices values and it stores the intervals into the output
   * columns. Regarding the interval of indices, a pair of values are taken from
   * the indices array in a consecutive manner. The pair of indices are left-closed
   * and right-open.
   *
   * The pairs of indices in the array are required to comply with the following
   * conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function returns an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The caller owns the output ColumnVectors and is responsible for closing them.
   *
   * @param indices
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public ColumnVector[] slice(int... indices) {
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "slice")) {
      long[] nativeHandles = slice(this.getNativeView(), indices);
      ColumnVector[] columnVectors = new ColumnVector[nativeHandles.length];
      for (int i = 0; i < nativeHandles.length; i++) {
        columnVectors[i] = new ColumnVector(nativeHandles[i]);
      }
      return columnVectors;
    }
  }

  /**
   * Return a subVector from start inclusive to the end of the vector.
   * @param start the index to start at.
   */
  public ColumnVector subVector(int start) {
    return subVector(start, (int)rows);
  }

  /**
   * Return a subVector.
   * @param start the index to start at (inclusive).
   * @param end the index to end at (exclusive).
   */
  public ColumnVector subVector(int start, int end) {
    ColumnVector [] tmp = slice(start, end);
    assert tmp.length == 1;
    return tmp[0];
  }

  /**
   * Splits a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them.
   *
   * The "split" function divides the input column into multiple intervals
   * of rows using the splits indices values and it stores the intervals into the
   * output columns. Regarding the interval of indices, a pair of values are taken
   * from the indices array in a consecutive manner. The pair of indices are
   * left-closed and right-open.
   *
   * The indices array ('splits') is require to be a monotonic non-decreasing set.
   * The indices in the array are required to comply with the following conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * The split function will take a pair of indices from the indices array
   * ('splits') in a consecutive manner. For the first pair, the function will
   * take the value 0 and the first element of the indices array. For the last pair,
   * the function will take the last element of the indices array and the size of
   * the input column.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function return an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The input columns may have different sizes. The number of
   * columns must be equal to the number of indices in the array plus one.
   *
   * Example:
   * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
   * splits: {2, 5, 9}
   * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
   *
   * Note that this is very similar to the output from a PartitionedTable.
   *
   * @param indices the indexes to split with
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public ColumnVector[] split(int... indices) {
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "split")) {
      long[] nativeHandles = split(this.getNativeView(), indices);
      ColumnVector[] columnVectors = new ColumnVector[nativeHandles.length];
      for (int i = 0; i < nativeHandles.length; i++) {
        columnVectors[i] = new ColumnVector(nativeHandles[i]);
      }
      return columnVectors;
    }
  }

  /**
   * Create a new vector of length rows, where each row is filled with the Scalar's
   * value
   * @param scalar - Scalar to use to fill rows
   * @param rows - Number of rows in the new ColumnVector
   * @return - new ColumnVector
   */
  public static ColumnVector fromScalar(Scalar scalar, int rows) {
    long amount = predictSizeFor(scalar.getType().sizeInBytes, rows, !scalar.isValid());
    try (DevicePrediction ignored = new DevicePrediction(amount, "fromScalar")) {
      long columnHandle = fromScalar(scalar.getScalarHandle(), rows);
      return new ColumnVector(columnHandle);
    }
  }

  /**
   * Create a new vector by concatenating multiple columns together.
   * Note that all columns must have the same type.
   */
  public static ColumnVector concatenate(ColumnVector... columns) {
    if (columns.length < 2) {
      throw new IllegalArgumentException("Concatenate requires 2 or more columns");
    }
    long total = 0;
    for (ColumnVector cv: columns) {
      total += cv.getDeviceMemorySize();
    }
    try (DevicePrediction prediction = new DevicePrediction(total, "concatenate")) {
      long[] columnHandles = new long[columns.length];
      for (int i = 0; i < columns.length; ++i) {
        columnHandles[i] = columns[i].getNativeView();
      }
      return new ColumnVector(concatenate(columnHandles));
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATE/TIME
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Get year from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector year() {
    assert type.isTimestamp();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "year")) {
      return new ColumnVector(year(getNativeView()));
    }
  }

  /**
   * Get month from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector month() {
    assert type.isTimestamp();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "month")) {
      return new ColumnVector(month(getNativeView()));
    }
  }

  /**
   * Get day from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector day() {
    assert type.isTimestamp();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "day")) {
      return new ColumnVector(day(getNativeView()));
    }
  }

  /**
   * Get hour from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector hour() {
    assert type.hasTimeResolution();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "hour")) {
      return new ColumnVector(hour(getNativeView()));
    }
  }

  /**
   * Get minute from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector minute() {
    assert type.hasTimeResolution();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "minute")) {
      return new ColumnVector(minute(getNativeView()));
    }
  }

  /**
   * Get second from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector second() {
    assert type.hasTimeResolution();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "second")) {
      return new ColumnVector(second(getNativeView()));
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // ARITHMETIC
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Transform a vector using a custom function.  Be careful this is not
   * simple to do.  You need to be positive you know what type of data you are
   * processing and how the data is laid out.  This also only works on fixed
   * length types.
   * @param udf This function will be applied to every element in the vector
   * @param isPtx is the code of the function ptx? true or C/C++ false.
   */
  public ColumnVector transform(String udf, boolean isPtx) {
    return new ColumnVector(transform(getNativeView(), udf, isPtx));
  }

  /**
   * Multiple different unary operations. The output is the same type as input.
   * @param op      the operation to perform
   * @return the result
   */
  public ColumnVector unaryOp(UnaryOp op) {
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "unaryOp")) {
      return new ColumnVector(unaryOperation(getNativeView(), op.nativeId));
    }
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
   * Calculate the hyperbolic sin, output is the same type as input.
   */
  public ColumnVector sinh() {
    return unaryOp(UnaryOp.SINH);
  }

  /**
   * Calculate the hyperbolic cos, output is the same type as input.
   */
  public ColumnVector cosh() {
    return unaryOp(UnaryOp.COSH);
  }

  /**
   * Calculate the hyperbolic tan, output is the same type as input.
   */
  public ColumnVector tanh() {
    return unaryOp(UnaryOp.TANH);
  }

  /**
   * Calculate the hyperbolic arcsin, output is the same type as input.
   */
  public ColumnVector arcsinh() {
    return unaryOp(UnaryOp.ARCSINH);
  }

  /**
   * Calculate the hyperbolic arccos, output is the same type as input.
   */
  public ColumnVector arccosh() {
    return unaryOp(UnaryOp.ARCCOSH);
  }

  /**
   * Calculate the hyperbolic arctan, output is the same type as input.
   */
  public ColumnVector arctanh() {
    return unaryOp(UnaryOp.ARCTANH);
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
   * Calculate the cube root, output is the same type as input.
   */
  public ColumnVector cbrt() {
    return unaryOp(UnaryOp.CBRT);
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
   * Rounds a floating-point argument to the closest integer value, but returns it as a float.
   */
  public ColumnVector rint() {
    return unaryOp(UnaryOp.RINT);
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
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(outType), "binaryOp")) {
      if (rhs instanceof ColumnVector) {
        ColumnVector cvRhs = (ColumnVector) rhs;
        assert rows == cvRhs.getRowCount();
        return new ColumnVector(binaryOp(this, cvRhs, op, outType));
      } else if (rhs instanceof Scalar) {
        Scalar sRhs = (Scalar) rhs;
        return new ColumnVector(binaryOp(this, sRhs, op, outType));
      } else {
        throw new IllegalArgumentException(rhs.getClass() + " is not supported as a binary op" +
            " with ColumnVector");
      }
    }
  }

  static long binaryOp(ColumnVector lhs, ColumnVector rhs, BinaryOp op, DType outputType) {
    return binaryOpVV(lhs.getNativeView(), rhs.getNativeView(),
        op.nativeId, outputType.nativeId);
  }

  static long binaryOp(ColumnVector lhs, Scalar rhs, BinaryOp op, DType outputType) {
    return binaryOpVS(lhs.getNativeView(), rhs.getScalarHandle(),
        op.nativeId, outputType.nativeId);
  }

  /////////////////////////////////////////////////////////////////////////////
  // AGGREGATION
  /////////////////////////////////////////////////////////////////////////////

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
    return reduce(AggregateOp.SUM, outType);
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
    return reduce(AggregateOp.MIN, outType);
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
    return reduce(AggregateOp.MAX, outType);
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
    return reduce(AggregateOp.PRODUCT, outType);
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
    return reduce(AggregateOp.SUMOFSQUARES, outType);
  }

  /**
   * Returns the arithmetic mean of all values in the column, returning a
   * FLOAT64 scalar unless the column type is FLOAT32 then a FLOAT32 scalar is returned.
   * Null values are skipped.
   */
  public Scalar mean() {
    DType outType = DType.FLOAT64;
    if (type == DType.FLOAT32) {
      outType = type;
    }
    return mean(outType);
  }

  /**
   * Returns the arithmetic mean of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
   */
  public Scalar mean(DType outType) {
    return reduce(AggregateOp.MEAN, outType);
  }

  /**
   * Returns the variance of all values in the column, returning a
   * FLOAT64 scalar unless the column type is FLOAT32 then a FLOAT32 scalar is returned.
   * Null values are skipped.
   */
  public Scalar variance() {
    DType outType = DType.FLOAT64;
    if (type == DType.FLOAT32) {
      outType = type;
    }
    return variance(outType);
  }

  /**
   * Returns the variance of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
   */
  public Scalar variance(DType outType) {
    return reduce(AggregateOp.VAR, outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a FLOAT64 scalar unless the column type is FLOAT32 then
   * a FLOAT32 scalar is returned. Nulls are not counted as an element
   * of the column when calculating the standard deviation.
   */
  public Scalar standardDeviation() {
    DType outType = DType.FLOAT64;
    if (type == DType.FLOAT32) {
      outType = type;
    }
    return standardDeviation(outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a scalar of the specified type. Null's are not counted as
   * an element of the column when calculating the standard deviation.
   */
  public Scalar standardDeviation(DType outType) {
    return reduce(AggregateOp.STD, outType);
  }

  /**
   * Returns a boolean scalar that is true if any of the elements in
   * the column are true or non-zero otherwise false.
   * Null values are skipped.
   */
  public Scalar any() {
    return any(DType.BOOL8);
  }

  /**
   * Returns a scalar is true or 1, depending on the specified type,
   * if any of the elements in the column are true or non-zero
   * otherwise false or 0.
   * Null values are skipped.
   */
  public Scalar any(DType outType) {
    return reduce(AggregateOp.ANY, outType);
  }

  /**
   * Returns a boolean scalar that is true if all of the elements in
   * the column are true or non-zero otherwise false.
   * Null values are skipped.
   */
  public Scalar all() {
    return all(DType.BOOL8);
  }

  /**
   * Returns a scalar is true or 1, depending on the specified type,
   * if all of the elements in the column are true or non-zero
   * otherwise false or 0.
   * Null values are skipped.
   */
  public Scalar all(DType outType) {
    return reduce(AggregateOp.ALL, outType);
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
  public Scalar reduce(AggregateOp op) {
    return reduce(op, type);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * supported for reduction of non-arithmetic types (TIMESTAMP...)
   * The null values are skipped for the operation.
   * @param op      The reduction operation to perform
   * @param outType The type of scalar value to return
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduce(AggregateOp op, DType outType) {
    return new Scalar(outType, reduce(getNativeView(), op.nativeId, outType.nativeId));
  }

  /**
   * Calculate the quantile of this ColumnVector
   * @param method   the method used to calculate the quantile
   * @param quantile the quantile value [0,1]
   * @return the quantile as double. The type can be changed in future
   */
  public Scalar quantile(QuantileMethod method, double quantile) {
    return new Scalar(type, quantile(getNativeView(), method.nativeId, quantile));
  }

  /**
   * This function aggregates values in a window around each element i of the input
   * column. Please refer to WindowsOptions for various options that can be passed.
   * @param op the operation to perform.
   * @param options various window function arguments.
   * @return Column containing aggregate function result.
   */
  public ColumnVector rollingWindow(AggregateOp op, WindowOptions options) {
    return new ColumnVector(
        rollingWindow(this.getNativeView(),
            options.getMinPeriods(),
            op.nativeId,
            options.getPreceding(),
            options.getFollowing(),
            options.getPrecedingCol() == null ? 0 : options.getPrecedingCol().getNativeView(),
            options.getFollowingCol() == null ? 0 : options.getFollowingCol().getNativeView()));
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
  // SEARCH
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Find if the `needle` is present in this col
   *
   * example:
   *
   *  Single Column:
   *      idx      0   1   2   3   4
   *      col = { 10, 20, 20, 30, 50 }
   *  Scalar:
   *   value = { 20 }
   *   result = true
   *
   * @param needle
   * @return true if needle is present else false
   */
  public boolean contains(Scalar needle) {
    return containsScalar(getNativeView(), needle.getScalarHandle());
  }

  /**
   * Returns a new ColumnVector of {@link DType#BOOL8} elements containing true if the corresponding
   * entry in haystack is contained in needles and false if it is not. The caller will be responsible
   * for the lifecycle of the new vector.
   *
   * example:
   *
   *   haystack = { 10, 20, 30, 40, 50 }
   *   needles  = { 20, 40, 60, 80 }
   *
   *   result = { false, true, false, true, false }
   *
   * @param needles
   * @return A new ColumnVector of type {@link DType#BOOL8}
   */
  public ColumnVector contains(ColumnVector needles) {
    return new ColumnVector(containsVector(getNativeView(), needles.getNativeView()));
  }

  /////////////////////////////////////////////////////////////////////////////
  // TYPE CAST
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Generic method to cast ColumnVector
   * When casting from a Date, Timestamp, or Boolean to a numerical type the underlying numerical
   * representation of the data will be used for the cast.
   *
   * For Strings:
   * Casting strings from/to timestamp isn't supported atm.
   * Please look at {@link ColumnVector#asTimestamp(DType, String)}
   * and {@link ColumnVector#asStrings(String)} for casting string to timestamp when the format
   * is known
   *
   * Float values when converted to String could be different from the expected default behavior in
   * Java
   * e.g.
   * 12.3 => "12.30000019" instead of "12.3"
   * Double.POSITIVE_INFINITY => "Inf" instead of "INFINITY"
   * Double.NEGATIVE_INFINITY => "-Inf" instead of "-INFINITY"
   *
   * @param type type of the resulting ColumnVector
   * @return A new vector allocated on the GPU
   */
  public ColumnVector castTo(DType type) {
    if (this.type == type) {
      // Optimization
      return incRefCount();
    }
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(type), "cast")) {
      return new ColumnVector(castTo(getNativeView(), type.nativeId));
    }
  }

  /**
   * Cast to Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asBytes() {
    return castTo(DType.INT8);
  }

  /**
   * Cast to Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asShorts() {
    return castTo(DType.INT16);
  }

  /**
   * Cast to Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asInts() {
    return castTo(DType.INT32);
  }

  /**
   * Cast to Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asLongs() {
    return castTo(DType.INT64);
  }

  /**
   * Cast to Float - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to float
   * When casting from a Date, Timestamp, or Boolean to a float type the underlying numerical
   * representatio of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asFloats() {
    return castTo(DType.FLOAT32);
  }

  /**
   * Cast to Double - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to double
   * When casting from a Date, Timestamp, or Boolean to a double type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asDoubles() {
    return castTo(DType.FLOAT64);
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampDays() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_DAYS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_DAYS);
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampDays(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_DAYS, format);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampSeconds() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_SECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_SECONDS);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampSeconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_SECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMicroseconds() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_MICROSECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_MICROSECONDS);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMicroseconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MICROSECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMilliseconds() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_MILLISECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_MILLISECONDS);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMilliseconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MILLISECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampNanoseconds() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_NANOSECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_NANOSECONDS);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampNanoseconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_NANOSECONDS, format);
  }

  /**
   * Parse a string to a timestamp. Strings that fail to parse will default to 0, corresponding
   * to 1970-01-01 00:00:00.000.
   * @param timestampType timestamp DType that includes the time unit to parse the timestamp into.
   * @param format strptime format specifier string of the timestamp. Used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z format specifiers.
   *               See https://github.com/rapidsai/custrings/blob/branch-0.10/docs/source/datetime.md
   *               for full parsing format specification and documentation.
   * @return A new ColumnVector containing the long representations of the timestamps in the
   *         original column vector.
   */
  public ColumnVector asTimestamp(DType timestampType, String format) {
    assert type == DType.STRING : "A column of type string " +
                                  "is required for .to_timestamp() operation";
    assert format != null : "Format string may not be NULL";
    assert timestampType.isTimestamp() : "unsupported conversion to non-timestamp DType";

    // Prediction could be better, but probably okay for now
    try (DevicePrediction prediction = new DevicePrediction(predictSizeForRowMult(format.length(), 2), "asTimestamp")) {
      return new ColumnVector(stringTimestampToTimestamp(getNativeView(),
          timestampType.nativeId, format));
    }
  }

  /**
   * Cast to Strings.
   * Negative timestamp values are not currently supported and will yield undesired results. See
   * github issue https://github.com/rapidsai/cudf/issues/3116 for details
   * In case of timestamps it follows the following formats
   *    {@link DType#TIMESTAMP_DAYS} - "%Y-%m-%d"
   *    {@link DType#TIMESTAMP_SECONDS} - "%Y-%m-%d %H:%M:%S"
   *    {@link DType#TIMESTAMP_MICROSECONDS} - "%Y-%m-%d %H:%M:%S.%f"
   *    {@link DType#TIMESTAMP_MILLISECONDS} - "%Y-%m-%d %H:%M:%S.%f"
   *    {@link DType#TIMESTAMP_NANOSECONDS} - "%Y-%m-%d %H:%M:%S.%f"
   *
   * @return A new vector allocated on the GPU.
   */
  public ColumnVector asStrings() {
    switch(type) {
      case TIMESTAMP_SECONDS:
        return asStrings("%Y-%m-%d %H:%M:%S");
      case TIMESTAMP_DAYS:
        return asStrings("%Y-%m-%d");
      case TIMESTAMP_MICROSECONDS:
      case TIMESTAMP_MILLISECONDS:
      case TIMESTAMP_NANOSECONDS:
        return asStrings("%Y-%m-%d %H:%M:%S.%f");
      default:
        return castTo(DType.STRING);
    }
  }

  /**
   * Method to parse and convert a timestamp column vector to string column vector. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00:000 in either
   * positive or negative direction.

   * No checking is done for invalid formats or invalid timestamp units.
   * Negative timestamp values are not currently supported and will yield undesired results. See
   * github issue https://github.com/rapidsai/cudf/issues/3116 for details
   *
   * @param format - strftime format specifier string of the timestamp. Its used to parse and convert
   *               the timestamp with. Supports %m,%j,%d,%H,%M,%S,%y,%Y,%f format specifiers.
   *               %d 	Day of the month: 01-31
   *               %m 	Month of the year: 01-12
   *               %y 	Year without century: 00-99c
   *               %Y 	Year with century: 0001-9999
   *               %H 	24-hour of the day: 00-23
   *               %M 	Minute of the hour: 00-59
   *               %S 	Second of the minute: 00-59
   *               %f 	6-digit microsecond: 000000-999999
   *               See https://github.com/rapidsai/custrings/blob/branch-0.10/docs/source/datetime.md
   *
   * Reported bugs
   * https://github.com/rapidsai/cudf/issues/4160 after the bug is fixed this method should
   * also support
   *               %I 	12-hour of the day: 01-12
   *               %p 	Only 'AM', 'PM'
   *               %j   day of the year
   *
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asStrings(String format) {
    assert type.isTimestamp() : "unsupported conversion from non-timestamp DType";
    assert format != null || format.isEmpty(): "Format string may not be NULL or empty";

    return new ColumnVector(timestampToStringTimestamp(this.getNativeView(), format));
  }

  /////////////////////////////////////////////////////////////////////////////
  // STRINGS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Convert a string to upper case.
   */
  public ColumnVector upper() {
    assert type == DType.STRING : "A column of type string is required for .upper() operation";
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "upper")) {
      return new ColumnVector(upperStrings(getNativeView()));
    }
  }

  /**
   * Convert a string to lower case.
   */
  public ColumnVector lower() {
    assert type == DType.STRING : "A column of type string is required for .lower() operation";
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "lower")) {
      return new ColumnVector(lowerStrings(getNativeView()));
    }
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column
   * into a single string row of a new column with no separator string inserted between each
   * combined string and maintaining null values in combined rows.
   * @param columns indefinite number of columns containing strings.
   * @return A new java column vector containing the concatenated strings.
   */
  public ColumnVector stringConcatenate(ColumnVector... columns) {
    try (Scalar emptyString = Scalar.fromString("");
         Scalar nullString = Scalar.fromString(null)) {
      return stringConcatenate(emptyString, nullString, columns);
    }
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column into
   * a single string row of a new column.
   * @param separator string scalar inserted between each string being merged.
   * @param narep string scalar indicating null behavior. If set to null and any string in the row
   *              is null the resulting string will be null. If not null, null values in any column
   *              will be replaced by the specified string.
   * @param columns indefinite number of columns containing strings, must be more than 2 columns
   * @return A new java column vector containing the concatenated strings.
   */
  public static ColumnVector stringConcatenate(Scalar separator, Scalar narep, ColumnVector... columns) {
    assert columns.length >= 2 : ".stringConcatenate() operation requires at least 2 columns";
    assert separator != null : "separator scalar provided may not be null";
    assert separator.getType() == DType.STRING : "separator scalar must be a string scalar";
    assert separator.isValid() == true : "separator string scalar may not contain a null value";
    assert narep != null : "narep scalar provided may not be null";
    assert narep.getType() == DType.STRING : "narep scalar must be a string scalar";
    long size = columns[0].getRowCount();
    long[] column_views = new long[columns.length];
    long deviceMemorySizeEstimate = 0;

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getType() == DType.STRING : "All columns must be of type string for .cat() operation";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must have the same number of rows";
      column_views[i] = columns[i].getNativeView();
      deviceMemorySizeEstimate += columns[i].getDeviceMemorySize();
    }

    try (DevicePrediction prediction = new DevicePrediction(deviceMemorySizeEstimate, "stringConcatenate")) {
      return new ColumnVector(stringConcatenation(column_views, separator.getScalarHandle(), narep.getScalarHandle()));
    }
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default values for start (0) and end index.
   * @param substring scalar containing the string to locate within each row.
   */
  public ColumnVector stringLocate(Scalar substring) {
    return stringLocate(substring, 0);
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default value for end index (-1, the end of each string).
   * @param substring scalar containing the string to locate within each row.
   * @param start character index to start the search from (inclusive).
   */
  public ColumnVector stringLocate(Scalar substring, int start) {
    return stringLocate(substring, start, -1);
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Can be be configured to start or end
   * the search mid string.
   * @param substring scalar containing the string scalar to locate within each row.
   * @param start character index to start the search from (inclusive).
   * @param end character index to end the search on (exclusive).
   */
  public ColumnVector stringLocate(Scalar substring, int start, int end) {
    assert type == DType.STRING : "column type must be a String";
    assert substring != null : "target string may not be null";
    assert substring.getType() == DType.STRING : "substring scalar must be a string scalar";
    assert substring.isValid() == true : "substring string scalar may not contain a null value";
    assert substring.getJavaString().isEmpty() == false : "substring string scalar may not be empty";
    assert start >= 0 : "start index must be a positive value";
    assert end >= start || end == -1 : "end index must be -1 or >= the start index";

    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT32), "stringLocate")) {
      return new ColumnVector(substringLocate(getNativeView(), substring.getScalarHandle(),
          start, end));
    }
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column.
   * Overloading subString to support if end index is not provided. Appending -1 to indicate to
   * read until end of string.
   * @param start first character index to begin the substring(inclusive).
   */
  public ColumnVector substring(int start) {
    return substring(start, -1);
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column.
   * 0-based indexing, If the stop position is past end of a string's length, then end of string is
   * used as stop position for that string.
   * @param start first character index to begin the substring(inclusive).
   * @param end   last character index to stop the substring(exclusive)
   * @return A new java column vector containing the substrings.
   */
  public ColumnVector substring(int start, int end) {
    assert type == DType.STRING : "column type must be a String";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT32), "subString")) {
      return new ColumnVector(substring(getNativeView(), start, end));
    }
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column
   * which uses unique ranges for each string
   * @param start Vector containing start indices of each string
   * @param end   Vector containing end indices of each string. -1 indicated to read until end of string.
   * @return A new java column vector containing the substrings/
   */
  public ColumnVector substring(ColumnVector start, ColumnVector end) {
    assert type == DType.STRING : "column type must be a String";
    assert (rows == start.getRowCount() && rows == end.getRowCount()) : "Number of rows must be equal";
    assert (start.getType() == DType.INT32 && end.getType() == DType.INT32) : "start and end " +
            "vectors must be of integer type";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT32), "subString")) {
      return new ColumnVector(substringColumn(getNativeView(), start.getNativeView(), end.getNativeView()));
    }
  }

  /**
   * Checks if each string in a column starts with a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param pattern scalar containing the string being searched for at the beginning of the column's strings.
   * @return A new java column vector containing the boolean results.
   */
  public ColumnVector startsWith(Scalar pattern) {
    assert type == DType.STRING : "column type must be a String";
    assert pattern != null : "pattern scalar may not be null";
    assert pattern.getType() == DType.STRING : "pattern scalar must be a string scalar";
    assert pattern.isValid() == true : "pattern string scalar may not contain a null value";
    assert pattern.getJavaString().isEmpty() == false : "pattern string scalar may not be empty";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8), "startsWith")) {
      return new ColumnVector(stringStartWith(getNativeView(), pattern.getScalarHandle()));
    }
  }

  /**
   * Checks if each string in a column ends with a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param pattern scalar containing the string being searched for at the end of the column's strings.
   * @return A new java column vector containing the boolean results.
   */
  public ColumnVector endsWith(Scalar pattern) {
    assert type == DType.STRING : "column type must be a String";
    assert pattern != null : "pattern scalar may not be null";
    assert pattern.getType() == DType.STRING : "pattern scalar must be a string scalar";
    assert pattern.isValid() == true : "pattern string scalar may not contain a null value";
    assert pattern.getJavaString().isEmpty() == false : "pattern string scalar may not be empty";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8), "endsWith")) {
      return new ColumnVector(stringEndWith(getNativeView(), pattern.getScalarHandle()));
    }
  }

  /**
   * Checks if each string in a column contains a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param compString scalar containing the string being searched for.
   * @return A new java column vector containing the boolean results.
   */

  public ColumnVector stringContains(Scalar compString) {
    assert type == DType.STRING : "column type must be a String";
    assert compString != null : "compString scalar may not be null";
    assert compString.getType() == DType.STRING : "compString scalar must be a string scalar";
    assert compString.isValid() : "compString string scalar may not contain a null value";
    assert !compString.getJavaString().isEmpty() : "compString string scalar may not be empty";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8), "stringContains")) {
      return new ColumnVector(stringContains(getNativeView(), compString.getScalarHandle()));
    }
  }

  /**
   * Returns a boolean ColumnVector identifying rows which
   * match the given regex pattern but only at the beginning of the string.
   *
   * ```
   * cv = ["abc","123","def456"]
   * result = cv.matches_re("\\d+")
   * r is now [false, true, false]
   * ```
   * Any null string entries return corresponding null output column entries.
   * For supported regex patterns refer to:
   * @link https://rapidsai.github.io/projects/nvstrings/en/0.13.0/regex.html
   *
   * @param pattern Regex pattern to match to each string.
   * @return New ColumnVector of boolean results for each string.
   */
  public ColumnVector matchesRe(String pattern) {
    assert type == DType.STRING : "column type must be a String";
    assert pattern != null : "pattern may not be null";
    assert pattern.isEmpty() == false : "pattern string may not be empty";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8), "matchesRe")) {
      return new ColumnVector(matchesRe(getNativeView(), pattern));
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  // INTERNAL/NATIVE ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * USE WITH CAUTION: This method exposes the address of the native cudf::column_view.  This allows
   * writing custom kernels or other cuda operations on the data.  DO NOT close this column
   * vector until you are completely done using the native column_view.  DO NOT modify the column in
   * any way.  This should be treated as a read only data structure. This API is unstable as
   * the underlying C/C++ API is still not stabilized.  If the underlying data structure
   * is renamed this API may be replaced.  The underlying data structure can change from release
   * to release (it is not stable yet) so be sure that your native code is complied against the
   * exact same version of libcudf as this is released for.
   */
  public long getNativeView() {
    return offHeap.getViewHandle();
  }

  /**
   * Native method to parse and convert a string column vector to unix timestamp. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00.000 in either
   * positive or negative direction. This mirrors the functionality spark sql's to_unix_timestamp.
   * Strings that fail to parse will default to 0. Supported time units are second, millisecond,
   * microsecond, and nanosecond. Larger time units for column vectors are not supported yet in cudf.
   * No checking is done for invalid formats or invalid timestamp units.
   * Negative timestamp values are not currently supported and will yield undesired results. See
   * github issue https://github.com/rapidsai/cudf/issues/3116 for details
   *
   * @param unit integer native ID of the time unit to parse the timestamp into.
   * @param format strptime format specifier string of the timestamp. Used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z format specifiers.
   *               See https://github.com/rapidsai/custrings/blob/branch-0.10/docs/source/datetime.md
   *               for full parsing format specification and documentation.
   * @return native handle of the resulting cudf column, used to construct the Java column vector
   *         by the timestampToLong method.
   */
  private static native long stringTimestampToTimestamp(long viewHandle, int unit, String format);

  /**
   * Native method to parse and convert a timestamp column vector to string column vector. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00:000 in either
   * positive or negative direction. This mirrors the functionality spark sql's from_unixtime.
   * No checking is done for invalid formats or invalid timestamp units.
   * Negative timestamp values are not currently supported and will yield undesired results. See
   * github issue https://github.com/rapidsai/cudf/issues/3116 for details
   *
   * @param format - strftime format specifier string of the timestamp. Its used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%M,%S,%f format specifiers.
   *               %d 	Day of the month: 01-31
   *               %m 	Month of the year: 01-12
   *               %y 	Year without century: 00-99c
   *               %Y 	Year with century: 0001-9999
   *               %H 	24-hour of the day: 00-23
   *               %M 	Minute of the hour: 00-59
   *               %S 	Second of the minute: 00-59
   *               %f 	6-digit microsecond: 000000-999999
   *               See http://man7.org/linux/man-pages/man3/strftime.3.html for details
   *
   * Reported bugs
   * https://github.com/rapidsai/cudf/issues/4160 after the bug is fixed this method should
   * also support
   *               %I 	12-hour of the day: 01-12
   *               %p 	Only 'AM', 'PM'
   *               %j   day of the year
   *
   * @return - native handle of the resulting cudf column used to construct the Java column vector
   */
  private static native long timestampToStringTimestamp(long viewHandle, String format);

  /**
   * Native method for locating the starting index of the first instance of a given substring
   * in each string in the column. 0 indexing, returns -1 if the substring is not found. Can be
   * be configured to start or end the search mid string.
   * @param columnView native handle of the cudf::column_view containing strings being operated on.
   * @param substringScalar string scalar handle containing the string to locate within each row.
   * @param start character index to start the search from (inclusive).
   * @param end character index to end the search on (exclusive).
   */
  private static native long substringLocate(long columnView, long substringScalar, int start, int end);

  /**
   * Native method to calculate substring from a given string column. 0 indexing.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param start      first character index to begin the substring(inclusive).
   * @param end        last character index to stop the substring(exclusive).
   */
  private static native long substring(long columnView, int start, int end) throws CudfException;

  /**
   * Native method to calculate substring from a given string column.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param startColumn handle of cudf::column_view which has start indices of each string.
   * @param endColumn handle of cudf::column_view which has end indices of each string.
   */
  private static native long substringColumn(long columnView, long startColumn, long endColumn)
          throws CudfException;
  /**
   * Native method for checking if strings in a column starts with a specified comparison string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param compString handle of scalar containing the string being searched for at the beginning
  of each string in the column.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long stringStartWith(long cudfViewHandle, long compString) throws CudfException;

  /**
   * Native method for checking if strings in a column ends with a specified comparison string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param compString handle of scalar containing the string being searched for at the end
  of each string in the column.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long stringEndWith(long cudfViewHandle, long compString) throws CudfException;

  private static native long matchesRe(long cudfViewHandle, String compString) throws CudfException;

  /**
   * Native method for checking if strings in a column contains a specified comparison string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param compString handle of scalar containing the string being searched for.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long stringContains(long cudfViewHandle, long compString) throws CudfException;

  /**
   * Native method to concatenate columns of strings together, combining a row from
   * each colunm into a single string.
   * @param columnViews array of longs holding the native handles of the column_views to combine.
   * @param separator string scalar inserted between each string being merged, may not be null.
   * @param narep string scalar indicating null behavior. If set to null and any string in the row is null
   *              the resulting string will be null. If not null, null values in any column will be
   *              replaced by the specified string. The underlying value in the string scalar may be null,
   *              but the object passed in may not.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the stringConcatenate method.
   */
  private static native long stringConcatenation(long[] columnViews, long separator, long narep);

  private static native long binaryOpVS(long lhs, long rhs, int op, int dtype);

  private static native long binaryOpVV(long lhs, long rhs, int op, int dtype);

  private static native long byteCount(long viewHandle) throws CudfException;

  private static native long castTo(long nativeHandle, int type);

  private static native long[] slice(long nativeHandle, int[] indices) throws CudfException;

  private static native long[] split(long nativeHandle, int[] indices) throws CudfException;

  private static native long findAndReplaceAll(long valuesHandle, long replaceHandle, long myself) throws CudfException;

  /**
   * Native method to switch all characters in a column of strings to lowercase characters.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the lower method.
   */
  private static native long lowerStrings(long cudfViewHandle);

  /**
   * Native method to switch all characters in a column of strings to uppercase characters.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the upper method.
   */
  private static native long upperStrings(long cudfViewHandle);

  private static native long quantile(long cudfColumnHandle, int quantileMethod, double quantile) throws CudfException;

  private static native long rollingWindow(long viewHandle, int min_periods, int agg_type,
                                           int preceding, int following,
                                           long preceding_col, long following_col);

  private static native long lengths(long viewHandle) throws CudfException;

  private static native long concatenate(long[] viewHandles) throws CudfException;

  private static native long fromScalar(long scalarHandle, int rowCount) throws CudfException;

  private static native long replaceNulls(long viewHandle, long scalarHandle) throws CudfException;

  private static native long ifElseVV(long predVec, long trueVec, long falseVec) throws CudfException;

  private static native long ifElseVS(long predVec, long trueVec, long falseScalar) throws CudfException;

  private static native long ifElseSV(long predVec, long trueScalar, long falseVec) throws CudfException;

  private static native long ifElseSS(long predVec, long trueScalar, long falseScalar) throws CudfException;

  private static native long reduce(long viewHandle, int reduceOp, int dtype) throws CudfException;

  private static native long isNullNative(long viewHandle);

  private static native long isNanNative(long viewHandle);

  private static native long isNotNanNative(long viewHandle);

  private static native long isNotNullNative(long viewHandle);

  private static native long unaryOperation(long viewHandle, int op);
  
  private static native long year(long viewHandle) throws CudfException;

  private static native long month(long viewHandle) throws CudfException;

  private static native long day(long viewHandle) throws CudfException;

  private static native long hour(long viewHandle) throws CudfException;

  private static native long minute(long viewHandle) throws CudfException;

  private static native long second(long viewHandle) throws CudfException;

  private static native boolean containsScalar(long columnViewHaystack, long scalarHandle) throws CudfException;

  private static native long containsVector(long columnViewHaystack, long columnViewNeedles) throws CudfException;
  
  private static native long transform(long viewHandle, String udf, boolean isPtx);

  /**
   * Get the number of bytes needed to allocate a validity buffer for the given number of rows.
   */
  static native long getNativeValidPointerSize(int size);

  ////////
  // Native cudf::column_view life cycle and metadata access methods. Life cycle methods
  // should typically only be called from the OffHeap inner class.
  ////////

  private static native int getNativeTypeId(long viewHandle) throws CudfException;

  private static native int getNativeRowCount(long viewHandle) throws CudfException;

  private static native int getNativeNullCount(long viewHandle) throws CudfException;

  private static native void deleteColumnView(long viewHandle) throws CudfException;

  private static native long[] getNativeDataPointer(long viewHandle) throws CudfException;

  private static native long[] getNativeOffsetsPointer(long viewHandle) throws CudfException;

  private static native long[] getNativeValidPointer(long viewHandle) throws CudfException;

  private static native long makeCudfColumnView(int type, long data, long dataSize, long offsets, long valid, int nullCount, int size);

  ////////
  // Native methods specific to cudf::column. These either take or create a cudf::column
  // instead of a cudf::column_view so they need to be used with caution. These should
  // only be called from the OffHeap inner class.
  ////////

  /**
   * Delete the column. This is not private because there are a few cases where Table
   * may get back an array of columns pointers and we want to have best effort in cleaning them up
   * on any failure.
   */
  static native void deleteCudfColumn(long cudfColumnHandle) throws CudfException;

  private static native int getNativeNullCountColumn(long cudfColumnHandle) throws CudfException;

  private static native void setNativeNullCountColumn(long cudfColumnHandle, int nullCount) throws CudfException;

  /**
   * Create a cudf::column_view from a cudf::column.
   * @param cudfColumnHandle the pointer to the cudf::column
   * @return a pointer to a cudf::column_view
   * @throws CudfException on any error
   */
  private static native long getNativeColumnView(long cudfColumnHandle) throws CudfException;

  private static native long makeEmptyCudfColumn(int type);

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  protected static final class OffHeapState extends MemoryCleaner.Cleaner {
    private final long internalId;
    // This must be kept in sync with the native code
    public static final long UNKNOWN_NULL_COUNT = -1;
    private long columnHandle;
    private long viewHandle = 0;
    private BaseDeviceMemoryBuffer data;
    private BaseDeviceMemoryBuffer valid;
    private BaseDeviceMemoryBuffer offsets;

    /**
     * Make a column form an existing cudf::column *.
     */
    public OffHeapState(long internalId, long columnHandle) {
      this.internalId = internalId;
      this.columnHandle = columnHandle;
      data = getNativeDataPointer();
      valid = getNativeValidPointer();
      offsets = getNativeOffsetsPointer();
    }

    /**
     * Create a cudf::column_view from device side data.
     */
    public OffHeapState(long internalId, DType type, int rows, Optional<Long> nullCount,
                        DeviceMemoryBuffer data, DeviceMemoryBuffer valid, DeviceMemoryBuffer offsets) {
      assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
          || !nullCount.isPresent();
      this.internalId = internalId;
      int nc = nullCount.orElse(UNKNOWN_NULL_COUNT).intValue();
      this.data = data;
      this.valid = valid;
      this.offsets = offsets;
      if (rows == 0) {
        this.columnHandle = makeEmptyCudfColumn(type.nativeId);
      } else {
        long cd = data == null ? 0 : data.address;
        long cdSize = data == null ? 0 : data.length;
        long od = offsets == null ? 0 : offsets.address;
        long vd = valid == null ? 0 : valid.address;
        this.viewHandle = makeCudfColumnView(type.nativeId, cd, cdSize, od, vd, nc, rows);
      }
    }

    /**
     * Create a cudf::column_view from contiguous device side data.
     */
    public OffHeapState(long internalId, long viewHandle, DeviceMemoryBuffer contiguousBuffer) {
      assert viewHandle != 0;
      this.internalId = internalId;
      this.viewHandle = viewHandle;

      data = contiguousBuffer.sliceFrom(getNativeDataPointer());
      valid = contiguousBuffer.sliceFrom(getNativeValidPointer());
      offsets = contiguousBuffer.sliceFrom(getNativeOffsetsPointer());
    }

    public long getViewHandle() {
      if (viewHandle == 0) {
        viewHandle = getNativeColumnView(columnHandle);
      }
      return viewHandle;
    }

    public long getNativeRowCount() {
      return ColumnVector.getNativeRowCount(getViewHandle());
    }

    public long getNativeNullCount() {
      if (viewHandle != 0) {
        return ColumnVector.getNativeNullCount(getViewHandle());
      }
      return getNativeNullCountColumn(columnHandle);
    }

    private void setNativeNullCount(int nullCount) throws CudfException {
      assert viewHandle == 0 : "Cannot set the null count if a view has already been created";
      assert columnHandle != 0;
      setNativeNullCountColumn(columnHandle, nullCount);
    }

    private DeviceMemoryBufferView getNativeValidPointer() {
      long[] values = ColumnVector.getNativeValidPointer(getViewHandle());
      if (values[0] == 0) {
        return null;
      }
      return new DeviceMemoryBufferView(values[0], values[1]);
    }

    private DeviceMemoryBufferView getNativeDataPointer() {
      long[] values = ColumnVector.getNativeDataPointer(getViewHandle());
      if (values[0] == 0) {
        return null;
      }
      return new DeviceMemoryBufferView(values[0], values[1]);
    }

    private DeviceMemoryBufferView getNativeOffsetsPointer() {
      long[] values = ColumnVector.getNativeOffsetsPointer(getViewHandle());
      if (values[0] == 0) {
        return null;
      }
      return new DeviceMemoryBufferView(values[0], values[1]);
    }

    public DType getNativeType() {
      return DType.fromNative(getNativeTypeId(getViewHandle()));
    }

    public BaseDeviceMemoryBuffer getData() {
      return data;
    }

    public BaseDeviceMemoryBuffer getValid() {
      return valid;
    }

    public BaseDeviceMemoryBuffer getOffsets() {
      return offsets;
    }

    @Override
    public void noWarnLeakExpected() {
      super.noWarnLeakExpected();
      if (valid != null) {
        valid.noWarnLeakExpected();
      }
      if (data != null) {
        data.noWarnLeakExpected();
      }
      if(offsets != null) {
        offsets.noWarnLeakExpected();
      }
    }

    @Override
    public String toString() {
      return String.valueOf(columnHandle == 0 ? viewHandle : columnHandle);
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      long size = getDeviceMemorySize();
      boolean neededCleanup = false;
      if (viewHandle != 0) {
        deleteColumnView(viewHandle);
        viewHandle = 0;
        neededCleanup = true;
      }
      if (columnHandle != 0) {
        deleteCudfColumn(columnHandle);
        columnHandle = 0;
        neededCleanup = true;
      }
      if (data != null) {
        data.close();
        data = null;
        neededCleanup = true;
      }
      if (valid != null) {
        valid.close();
        valid = null;
        neededCleanup = true;
      }
      if (offsets != null) {
        offsets.close();
        offsets = null;
        neededCleanup = true;
      }
      if (neededCleanup) {
        if (logErrorIfNotClean) {
          log.error("YOU LEAKED A DEVICE COLUMN VECTOR!!!!");
          logRefCountDebug("Leaked vector");
        }
        MemoryListener.deviceDeallocation(size, internalId);
      }
      return neededCleanup;
    }

    /**
     * This returns total memory allocated in device for the ColumnVector.
     * @return number of device bytes allocated for this column
     */
    public long getDeviceMemorySize() {
      long size = valid != null ? valid.getLength() : 0;
      size += offsets != null ? offsets.getLength() : 0;
      size += data != null ? data.getLength() : 0;
      return size;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // BUILDER
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Create a new vector.
   * @param type       the type of vector to build.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static ColumnVector build(DType type, int rows, Consumer<Builder> init) {
    try (Builder builder = HostColumnVector.builder(type, rows)) {
      init.accept(builder);
      return builder.buildAndPutOnDevice();
    }
  }

  public static ColumnVector build(int rows, long stringBufferSize, Consumer<Builder> init) {
    try (Builder builder = HostColumnVector.builder(rows, stringBufferSize)) {
      init.accept(builder);
      return builder.buildAndPutOnDevice();
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
  public static ColumnVector daysFromInts(int... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampMilliSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampMicroSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampNanoSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls. This is really intended to be used only for testing as
   * it is slow and memory intensive to translate between java strings and UTF8 strings.
   */
  public static ColumnVector fromStrings(String... values) {
    try (HostColumnVector host = HostColumnVector.fromStrings(values)) {
      return host.copyToDevice();
    }
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
  public static ColumnVector timestampDaysFromBoxedInts(Integer... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampMilliSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampMicroSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampNanoSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
  }
}
