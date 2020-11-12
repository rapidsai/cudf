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
import ai.rapids.cudf.WindowOptions.FrameType;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * This class represents the immutable vector of data.  This class holds
 * references to device(GPU) memory and is reference counted to know when to release it.  Call
 * close to decrement the reference count when you are done with the column, and call incRefCount
 * to increment the reference count.
 */
public final class ColumnVector implements AutoCloseable, BinaryOperable {
  private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private ColumnView columnView;
  private final DType type;
  private final OffHeapState offHeap;
  private final long rows;
  private Optional<Long> nullCount = Optional.empty();
  private int refCount;

  /**
   * Wrap an existing on device cudf::column with the corresponding ColumnVector. The new
   * ColumnVector takes ownership of the pointer and will free it when the ref count reaches zero.
   * @param nativePointer host address of the cudf::column object which will be
   *                      owned by this instance.
   */
  public ColumnVector(long nativePointer) {
    assert nativePointer != 0;
    offHeap = new OffHeapState(nativePointer);
    MemoryCleaner.register(this, offHeap);
    columnView = new ColumnView(offHeap.viewHandle);
    this.rows = offHeap.getNativeRowCount();
    this.type = offHeap.getNativeType();
    this.refCount = 0;
    incRefCountInternal(true);
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
    assert type != DType.LIST : "This constructor should not be used for list type";
    if (type != DType.STRING) {
      assert offsetBuffer == null : "offsets are only supported for STRING";
    }

    long[] children = new long[] {};
    offHeap = new OffHeapState(type, (int) rows, nullCount, dataBuffer, validityBuffer, offsetBuffer, null, children);
    MemoryCleaner.register(this, offHeap);
    columnView = new ColumnView(offHeap.getViewHandle());
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;

    this.refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * Create a new column vector based off of data already on the device with child columns.
   * @param type the type of the vector, typically a nested type
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
   * @param nestedColumnVectors the child columns list
   */
  public ColumnVector(DType type, long rows, Optional<Long> nullCount,
                      BaseDeviceMemoryBuffer dataBuffer, BaseDeviceMemoryBuffer validityBuffer,
                      BaseDeviceMemoryBuffer offsetBuffer, List<ColumnView.NestedColumnVector> nestedColumnVectors) {
    if (type != DType.STRING && type != DType.LIST) {
      assert offsetBuffer == null : "offsets are only supported for STRING, LISTS";
    }
    List<BaseDeviceMemoryBuffer> toClose = new ArrayList<>();
    long[] childHandles = new long[nestedColumnVectors.size()];
    for (ColumnView.NestedColumnVector ncv : nestedColumnVectors) {
      toClose.addAll(ncv.getBuffersToClose());
    }
    for (int i = 0; i < nestedColumnVectors.size(); i++) {
      childHandles[i] = nestedColumnVectors.get(i).getViewHandle();
    }
    offHeap = new OffHeapState(type, (int) rows, nullCount, dataBuffer, validityBuffer, offsetBuffer,
            toClose, childHandles);
    MemoryCleaner.register(this, offHeap);
    columnView = new ColumnView(offHeap.getViewHandle());
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;

    this.refCount = 0;
    incRefCountInternal(true);
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
    offHeap = new OffHeapState(viewAddress, contiguousBuffer);
    MemoryCleaner.register(this, offHeap);
    //TODO init column view
    this.type = offHeap.getNativeType();
    this.rows = offHeap.getNativeRowCount();
    // TODO we may want to ask for the null count anyways...
    this.nullCount = Optional.empty();

    this.refCount = 0;
    incRefCountInternal(true);
  }

  static ColumnVector fromViewWithContiguousAllocation(long columnViewAddress, DeviceMemoryBuffer buffer) {
    return new ColumnVector(columnViewAddress, buffer);
  }

  /**
   * Returns a column of strings where, for each string row in the input,
   * the first character after spaces is modified to upper-case,
   * while all the remaining characters in a word are modified to lower-case.
   *
   * Any null string entries return corresponding null output column entries
   */
  public ColumnVector toTitle() {
    assert type == DType.STRING;
    return new ColumnVector(columnView.title(getNativeView()));
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
      offHeap.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times " + this);
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
   * Returns a new ColumnVector with NaNs converted to nulls, preserving the existing null values.
   */
  public ColumnVector nansToNulls() {
    return columnView.nansToNulls();
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
  public ColumnVector getCharLengths() {
    return columnView.getCharLengths();
  }

  /**
   * Retrieve the number of bytes for each string. Null strings will have value of null.
   *
   * @return ColumnVector, where each element at i = byte count of string at index 'i' in the original vector
   */
  public ColumnVector getByteCount() {
    return columnView.getByteCount();
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is not null, and FALSE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public ColumnVector isNotNull() {
    return columnView.isNotNull();
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * FALSE for any entry that is not null, and TRUE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public ColumnVector isNull() {
    return columnView.isNull();
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is an integer, and FALSE if its not an integer. A null will be returned
   * for null entries
   *
   * NOTE: Integer doesn't mean a 32-bit integer. It means a number that is not a fraction.
   * i.e. If this method returns true for a value it could still result in an overflow or underflow
   * if you convert it to a Java integral type
   *
   * @return - Boolean vector
   */
  public ColumnVector isInteger() {
    return columnView.isInteger();
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is a float, and FALSE if its not a float. A null will be returned
   * for null entries
   *
   * NOTE: Float doesn't mean a 32-bit float. It means a number that is a fraction or can be written
   * as a fraction. i.e. This method will return true for integers as well as floats. Also note if
   * this method returns true for a value it could still result in an overflow or underflow if you
   * convert it to a Java float or double
   *
   * @return - Boolean vector
   */
  public ColumnVector isFloat() {
    return columnView.isFloat();
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is NaN, and FALSE if null or a valid floating point value
   * @return - Boolean vector
   */
  public ColumnVector isNan() {
    return columnView.isNan();
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is null or a valid floating point value, FALSE otherwise
   * @return - Boolean vector
   */
  public ColumnVector isNotNan() {
    return columnView.isNotNan();
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
    return columnView.findAndReplaceAll(oldValues, newValues);
  }

  /**
   * Returns a ColumnVector with any null values replaced with a scalar.
   * The types of the input ColumnVector and Scalar must match, else an error is thrown.
   *
   * @param scalar - Scalar value to use as replacement
   * @return - ColumnVector with nulls replaced by scalar
   */
  public ColumnVector replaceNulls(Scalar scalar) {
    return columnView.replaceNulls(scalar);
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
    return columnView.ifElse(trueValues, falseValues);
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
    return columnView.ifElse(trueValues, falseValue);
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
    return columnView.ifElse(trueValue, falseValues);
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
    return columnView.ifElse(trueValue, falseValue);
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
    return columnView.slice(indices);
  }

  /**
   * Return a subVector from start inclusive to the end of the vector.
   * @param start the index to start at.
   */
  public ColumnVector subVector(int start) {
    return columnView.subVector(start);
  }

  /**
   * Return a subVector.
   * @param start the index to start at (inclusive).
   * @param end the index to end at (exclusive).
   */
  public ColumnVector subVector(int start, int end) {
    return columnView.subVector(start, end);
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
    return columnView.split(indices);
  }

  /**
   * Create a new vector of length rows, where each row is filled with the Scalar's
   * value
   * @param scalar - Scalar to use to fill rows
   * @param rows - Number of rows in the new ColumnVector
   * @return - new ColumnVector
   */
  public static ColumnVector fromScalar(Scalar scalar, int rows) {
    return ColumnView.fromScalar(scalar, rows);
  }

  /**
   * Create a new vector of length rows, starting at the initialValue and going by step each time.
   * Only numeric types are supported.
   * @param initialValue the initial value to start at.
   * @param step the step to add to each subsequent row.
   * @param rows the total number of rows
   * @return the new ColumnVector.
   */
  public static ColumnVector sequence(Scalar initialValue, Scalar step, int rows) {
    return ColumnView.sequence(initialValue, step, rows);
  }

  /**
   * Create a new vector of length rows, starting at the initialValue and going by 1 each time.
   * Only numeric types are supported.
   * @param initialValue the initial value to start at.
   * @param rows the total number of rows
   * @return the new ColumnVector.
   */
  public static ColumnVector sequence(Scalar initialValue, int rows) {
    return ColumnView.sequence(initialValue, rows);
  }

  /**
   * Create a new vector by concatenating multiple columns together.
   * Note that all columns must have the same type.
   */
  public static ColumnVector concatenate(ColumnVector... columns) {
    return ColumnView.concatenate(columns);
  }

  /**
   * Create a new vector of "normalized" values, where:
   *  1. All representations of NaN (and -NaN) are replaced with the normalized NaN value
   *  2. All elements equivalent to 0.0 (including +0.0 and -0.0) are replaced with +0.0.
   *  3. All elements that are not equivalent to NaN or 0.0 remain unchanged.
   *
   * The documentation for {@link Double#longBitsToDouble(long)}
   * describes how equivalent values of NaN/-NaN might have different bitwise representations.
   *
   * This method may be used to compare different bitwise values of 0.0 or NaN as logically
   * equivalent. For instance, if these values appear in a groupby key column, without normalization
   * 0.0 and -0.0 would be erroneously treated as distinct groups, as will each representation of NaN.
   *
   * @return A new ColumnVector with all elements equivalent to NaN/0.0 replaced with a normalized equivalent.
   */
  public ColumnVector normalizeNANsAndZeros() {
    return columnView.normalizeNANsAndZeros();
  }

  /**
   * Create a deep copy of the column while replacing the null mask. The resultant null mask is the
   * bitwise merge of null masks in the columns given as arguments.
   *
   * @param mergeOp binary operator, currently only BITWISE_AND is supported.
   * @param columns array of columns whose null masks are merged, must have identical number of rows.
   * @return the new ColumnVector with merged null mask.
   */
  public ColumnVector mergeAndSetValidity(BinaryOp mergeOp, ColumnVector... columns) {
    return columnView.mergeAndSetValidity(mergeOp, columns);
  }

  /**
   * Create a new vector containing the MD5 hash of each row in the table.
   *
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32 character hex strings representing each row's hash value.
   */
  public static ColumnVector md5Hash(ColumnVector... columns) {
    return ColumnView.md5Hash(columns);
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
    return columnView.year();
  }

  /**
   * Get month from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector month() {
    return columnView.month();
  }

  /**
   * Get day from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector day() {
    return columnView.day();
  }

  /**
   * Get hour from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector hour() {
    return columnView.hour();
  }

  /**
   * Get minute from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector minute() {
    return columnView.minute();
  }

  /**
   * Get second from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU.
   */
  public ColumnVector second() {
    return columnView.second();
  }

  /**
   * Get the day of the week from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU. Monday=1, ..., Sunday=7
   */
  public ColumnVector weekDay() {
    return columnView.weekDay();
  }

  /**
   * Get the date that is the last day of the month for this timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new TIMESTAMP_DAYS vector allocated on the GPU.
   */
  public ColumnVector lastDayOfMonth() {
    return columnView.lastDayOfMonth();
  }

  /**
   * Get the day of the year from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU. The value is between [1, {365-366}]
   */
  public ColumnVector dayOfYear() {
    return columnView.dayOfYear();
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
    return columnView.transform(udf, isPtx);
  }

  /**
   * Multiple different unary operations. The output is the same type as input.
   * @param op      the operation to perform
   * @return the result
   */
  public ColumnVector unaryOp(UnaryOp op) {
    return columnView.unaryOp(op);
  }

  /**
   * Calculate the sin, output is the same type as input.
   */
  public ColumnVector sin() {
    return columnView.sin();
  }

  /**
   * Calculate the cos, output is the same type as input.
   */
  public ColumnVector cos() {
    return columnView.cos();
  }

  /**
   * Calculate the tan, output is the same type as input.
   */
  public ColumnVector tan() {
    return columnView.tan();
  }

  /**
   * Calculate the arcsin, output is the same type as input.
   */
  public ColumnVector arcsin() {
    return columnView.arcsin();
  }

  /**
   * Calculate the arccos, output is the same type as input.
   */
  public ColumnVector arccos() {
    return columnView.arccos();
  }

  /**
   * Calculate the arctan, output is the same type as input.
   */
  public ColumnVector arctan() {
    return columnView.arctan();
  }

  /**
   * Calculate the hyperbolic sin, output is the same type as input.
   */
  public ColumnVector sinh() {
    return columnView.sinh();
  }

  /**
   * Calculate the hyperbolic cos, output is the same type as input.
   */
  public ColumnVector cosh() {
    return columnView.cosh();
  }

  /**
   * Calculate the hyperbolic tan, output is the same type as input.
   */
  public ColumnVector tanh() {
    return columnView.tanh();
  }

  /**
   * Calculate the hyperbolic arcsin, output is the same type as input.
   */
  public ColumnVector arcsinh() {
    return columnView.arcsinh();
  }

  /**
   * Calculate the hyperbolic arccos, output is the same type as input.
   */
  public ColumnVector arccosh() {
    return columnView.arccosh();
  }

  /**
   * Calculate the hyperbolic arctan, output is the same type as input.
   */
  public ColumnVector arctanh() {
    return columnView.arctanh();
  }

  /**
   * Calculate the exp, output is the same type as input.
   */
  public ColumnVector exp() {
    return columnView.exp();
  }

  /**
   * Calculate the log, output is the same type as input.
   */
  public ColumnVector log() {
    return columnView.log();
  }

  /**
   * Calculate the log with base 2, output is the same type as input.
   */
  public ColumnVector log2() {
    return columnView.log2();
  }

  /**
   * Calculate the log with base 10, output is the same type as input.
   */
  public ColumnVector log10() {
    return columnView.log10();
  }

  /**
   * Calculate the sqrt, output is the same type as input.
   */
  public ColumnVector sqrt() {
    return columnView.sqrt();
  }

  /**
   * Calculate the cube root, output is the same type as input.
   */
  public ColumnVector cbrt() {
    return columnView.cbrt();
  }

  /**
   * Calculate the ceil, output is the same type as input.
   */
  public ColumnVector ceil() {
    return columnView.ceil();
  }

  /**
   * Calculate the floor, output is the same type as input.
   */
  public ColumnVector floor() {
    return columnView.floor();
  }

  /**
   * Calculate the abs, output is the same type as input.
   */
  public ColumnVector abs() {
    return columnView.abs();
  }

  /**
   * Rounds a floating-point argument to the closest integer value, but returns it as a float.
   */
  public ColumnVector rint() {
    return columnView.rint();
  }

  /**
   * invert the bits, output is the same type as input.
   */
  public ColumnVector bitInvert() {
    return columnView.bitInvert();
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
      return columnView.binaryOp(op, ((ColumnVector) rhs).columnView, outType);
    } else {
      return columnView.binaryOp(op, rhs, outType);
    }
  }

  static long binaryOp(ColumnVector lhs, ColumnVector rhs, BinaryOp op, DType outputType) {
    return ColumnView.binaryOp(lhs.columnView, rhs.columnView, op, outputType);
  }

  static long binaryOp(ColumnVector lhs, Scalar rhs, BinaryOp op, DType outputType) {
    return ColumnView.binaryOp(lhs.columnView, rhs, op, outputType);
  }

  /////////////////////////////////////////////////////////////////////////////
  // AGGREGATION
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar sum() {
    return columnView.sum();
  }

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar sum(DType outType) {
    return columnView.sum(outType);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar min() {
    return columnView.min();
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar min(DType outType) {
    return columnView.min(outType);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar max() {
    return columnView.max();
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar max(DType outType) {
    return columnView.max(outType);
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar product() {
    return columnView.product();
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar product(DType outType) {
    return columnView.product(outType);
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the same type as this column.
   */
  public Scalar sumOfSquares() {
    return columnView.sumOfSquares();
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the specified type.
   */
  public Scalar sumOfSquares(DType outType) {
    return columnView.sumOfSquares(outType);
  }

  /**
   * Returns the arithmetic mean of all values in the column, returning a
   * FLOAT64 scalar unless the column type is FLOAT32 then a FLOAT32 scalar is returned.
   * Null values are skipped.
   */
  public Scalar mean() {
    return columnView.mean();
  }

  /**
   * Returns the arithmetic mean of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
   */
  public Scalar mean(DType outType) {
    return columnView.mean(outType);
  }

  /**
   * Returns the variance of all values in the column, returning a
   * FLOAT64 scalar unless the column type is FLOAT32 then a FLOAT32 scalar is returned.
   * Null values are skipped.
   */
  public Scalar variance() {
    return columnView.variance();
  }

  /**
   * Returns the variance of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
   */
  public Scalar variance(DType outType) {
    return columnView.variance(outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a FLOAT64 scalar unless the column type is FLOAT32 then
   * a FLOAT32 scalar is returned. Nulls are not counted as an element
   * of the column when calculating the standard deviation.
   */
  public Scalar standardDeviation() {
    return columnView.standardDeviation();
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a scalar of the specified type. Null's are not counted as
   * an element of the column when calculating the standard deviation.
   */
  public Scalar standardDeviation(DType outType) {
    return columnView.standardDeviation(outType);
  }

  /**
   * Returns a boolean scalar that is true if any of the elements in
   * the column are true or non-zero otherwise false.
   * Null values are skipped.
   */
  public Scalar any() {
    return columnView.any();
  }

  /**
   * Returns a scalar is true or 1, depending on the specified type,
   * if any of the elements in the column are true or non-zero
   * otherwise false or 0.
   * Null values are skipped.
   */
  public Scalar any(DType outType) {
    return columnView.any(outType);
  }

  /**
   * Returns a boolean scalar that is true if all of the elements in
   * the column are true or non-zero otherwise false.
   * Null values are skipped.
   */
  public Scalar all() {
    return columnView.all();
  }

  /**
   * Returns a scalar is true or 1, depending on the specified type,
   * if all of the elements in the column are true or non-zero
   * otherwise false or 0.
   * Null values are skipped.
   */
  public Scalar all(DType outType) {
    return columnView.all(outType);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * The null values are skipped for the operation.
   * @param aggregation The reduction aggregation to perform
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduce(Aggregation aggregation) {
    return columnView.reduce(aggregation);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * supported for reduction of non-arithmetic types (TIMESTAMP...)
   * The null values are skipped for the operation.
   * @param aggregation The reduction aggregation to perform
   * @param outType The type of scalar value to return
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduce(Aggregation aggregation, DType outType) {
    return columnView.reduce(aggregation, outType);
  }

  /**
   * Calculate various quantiles of this ColumnVector.  It is assumed that this is already sorted
   * in the desired order.
   * @param method   the method used to calculate the quantiles
   * @param quantiles the quantile values [0,1]
   * @return the quantiles as doubles, in the same order passed in. The type can be changed in future
   */
  public ColumnVector quantile(QuantileMethod method, double[] quantiles) {
    return columnView.quantile(method, quantiles);
  }

  /**
   * This function aggregates values in a window around each element i of the input
   * column. Please refer to WindowsOptions for various options that can be passed.
   * Note: Only rows-based windows are supported.
   * @param op the operation to perform.
   * @param options various window function arguments.
   * @return Column containing aggregate function result.
   * @throws IllegalArgumentException if unsupported window specification * (i.e. other than {@link FrameType#ROWS} is used.
   */
  public ColumnVector rollingWindow(Aggregation op, WindowOptions options) {
    return columnView.rollingWindow(op, options);
  }

  /////////////////////////////////////////////////////////////////////////////
  // LOGICAL
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Returns a vector of the logical `not` of each value in the input
   * column (this)
   */
  public ColumnVector not() {
    return columnView.not();
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
    return columnView.contains(needle);
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
    return columnView.contains(needles);
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
    return columnView.castTo(type);
  }

  /**
   * Cast to Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asBytes() {
    return columnView.asBytes();
  }

  /**
   * Cast to list of bytes
   * This method converts the rows provided by the ColumnVector and casts each row to a list of
   * bytes with endinanness reversed. Numeric and string types supported, but not timestamps.
   *
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asByteList() {
    return columnView.asByteList();
  }

  /**
   * Cast to list of bytes
   * This method converts the rows provided by the ColumnVector and casts each row to a list
   * of bytes. Numeric and string types supported, but not timestamps.
   *
   * @param config Flips the byte order (endianness) if true, retains byte order otherwise
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asByteList(boolean config) {
    return columnView.asByteList(config);
  }

  /**
   * Cast to unsigned Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned byte type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Byte#toUnsignedInt(byte)}.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asUnsignedBytes() {
    return columnView.asUnsignedBytes();
  }

  /**
   * Cast to Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asShorts() {
    return columnView.asShorts();
  }

  /**
   * Cast to unsigned Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned short type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Short#toUnsignedInt(short)}.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asUnsignedShorts() {
    return columnView.asUnsignedShorts();
  }

  /**
   * Cast to Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asInts() {
    return columnView.asInts();
  }

  /**
   * Cast to unsigned Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned int type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Integer#toUnsignedLong(int)}.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asUnsignedInts() {
    return columnView.asUnsignedInts();
  }

  /**
   * Cast to Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asLongs() {
    return columnView.asLongs();
  }

  /**
   * Cast to unsigned Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned long type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Long#toUnsignedString(long)}.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asUnsignedLongs() {
    return columnView.asUnsignedLongs();
  }

  /**
   * Cast to Float - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to float
   * When casting from a Date, Timestamp, or Boolean to a float type the underlying numerical
   * representatio of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asFloats() {
    return columnView.asFloats();
  }

  /**
   * Cast to Double - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to double
   * When casting from a Date, Timestamp, or Boolean to a double type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asDoubles() {
    return columnView.asDoubles();
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampDays() {
    return columnView.asTimestampDays();
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampDays(String format) {
    return columnView.asTimestampDays(format);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampSeconds() {
    return columnView.asTimestampSeconds();
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampSeconds(String format) {
    return columnView.asTimestampSeconds(format);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMicroseconds() {
    return columnView.asTimestampMicroseconds();
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMicroseconds(String format) {
    return asTimestampMicroseconds(format);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMilliseconds() {
   return columnView.asTimestampMilliseconds();
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMilliseconds(String format) {
    return asTimestampMilliseconds(format);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampNanoseconds() {
    return columnView.asTimestampNanoseconds();
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampNanoseconds(String format) {
   return asTimestampNanoseconds(format);
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
    return columnView.asTimestamp(timestampType, format);
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
    return columnView.asStrings();
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
    return columnView.asStrings(format);
  }

  /////////////////////////////////////////////////////////////////////////////
  // LISTS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * For each list in this column pull out the entry at the given index. If the entry would
   * go off the end of the list a NULL is returned instead.
   * @param index 0 based offset into the list. Negative values go backwards from the end of the
   *              list.
   * @return a new column of the values at those indexes.
   */
  public ColumnVector extractListElement(int index) {
    return columnView.extractListElement(index);
  }

  /////////////////////////////////////////////////////////////////////////////
  // STRINGS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Convert a string to upper case.
   */
  public ColumnVector upper() {
    return columnView.upper();
  }

  /**
   * Convert a string to lower case.
   */
  public ColumnVector lower() {
    return columnView.lower();
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column
   * into a single string row of a new column with no separator string inserted between each
   * combined string and maintaining null values in combined rows.
   * @param columns array of columns containing strings.
   * @return A new java column vector containing the concatenated strings.
   */
  public ColumnVector stringConcatenate(ColumnVector[] columns) {
    return columnView.stringConcatenate(columns);
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column into
   * a single string row of a new column.
   * @param separator string scalar inserted between each string being merged.
   * @param narep string scalar indicating null behavior. If set to null and any string in the row
   *              is null the resulting string will be null. If not null, null values in any column
   *              will be replaced by the specified string.
   * @param columns array of columns containing strings, must be more than 2 columns
   * @return A new java column vector containing the concatenated strings.
   */
  public static ColumnVector stringConcatenate(Scalar separator, Scalar narep, ColumnVector[] columns) {
    return ColumnView.stringConcatenate(separator, narep, columns);
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default values for start (0) and end index.
   * @param substring scalar containing the string to locate within each row.
   */
  public ColumnVector stringLocate(Scalar substring) {
    return columnView.stringLocate(substring);
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default value for end index (-1, the end of each string).
   * @param substring scalar containing the string to locate within each row.
   * @param start character index to start the search from (inclusive).
   */
  public ColumnVector stringLocate(Scalar substring, int start) {
    return columnView.stringLocate(substring, start);
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
    return columnView.stringLocate(substring, start, end);
  }

  /**
   * Returns a list of columns by splitting each string using the specified delimiter.
   * The number of rows in the output columns will be the same as the input column.
   * Null entries are added for a row where split results have been exhausted.
   * Null string entries return corresponding null output columns.
   * @param delimiter UTF-8 encoded string identifying the split points in each string.
   *                  An empty string indicates split on whitespace.
   * @return New table of strings columns.
   */
  public Table stringSplit(Scalar delimiter) {
    return columnView.stringSplit(delimiter);
  }

  /**
   * Returns a list of columns by splitting each string using whitespace as the delimiter.
   * The number of rows in the output columns will be the same as the input column.
   * Null entries are added for a row where split results have been exhausted.
   * Null string entries return corresponding null output columns.
   * @return New table of strings columns.
   */
  public Table stringSplit() {
    return columnView.stringSplit();
  }

  /**
   * Returns a column of lists of strings by splitting each string using whitespace as the delimiter.
   */
  public ColumnVector stringSplitRecord() {
    return columnView.stringSplitRecord();
  }

  /**
   * Returns a column of lists of strings by splitting each string using whitespace as the delimiter.
   * @param maxSplit the maximum number of records to split, or -1 for all of them.
   */
  public ColumnVector stringSplitRecord(int maxSplit) {
    return columnView.stringSplitRecord(maxSplit);
  }

  /**
   * Returns a column of lists of strings by splitting each string using the specified delimiter.
   * @param delimiter UTF-8 encoded string identifying the split points in each string.
   *                  An empty string indicates split on whitespace.
   */
  public ColumnVector stringSplitRecord(Scalar delimiter) {
    return columnView.stringSplitRecord(delimiter);
  }

  /**
   * Returns a column that is a list of strings. Each string list is made by splitting each input
   * string using the specified delimiter.
   * @param delimiter UTF-8 encoded string identifying the split points in each string.
   *                  An empty string indicates split on whitespace.
   * @param maxSplit the maximum number of records to split, or -1 for all of them.
   * @return New table of strings columns.
   */
  public ColumnVector stringSplitRecord(Scalar delimiter, int maxSplit) {
    return columnView.stringSplitRecord(delimiter, maxSplit);
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column.
   * Overloading subString to support if end index is not provided. Appending -1 to indicate to
   * read until end of string.
   * @param start first character index to begin the substring(inclusive).
   */
  public ColumnVector substring(int start) {
    return columnView.substring(start);
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
    return columnView.substring(start, end);
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column
   * which uses unique ranges for each string
   * @param start Vector containing start indices of each string
   * @param end   Vector containing end indices of each string. -1 indicated to read until end of string.
   * @return A new java column vector containing the substrings/
   */
  public ColumnVector substring(ColumnVector start, ColumnVector end) {
    return columnView.substring(start, end);
  }

  /**
   * Returns a new strings column where target string within each string is replaced with the specified
   * replacement string.
   * The replacement proceeds from the beginning of the string to the end, for example,
   * replacing "aa" with "b" in the string "aaa" will result in "ba" rather than "ab".
   * Specifing an empty string for replace will essentially remove the target string if found in each string.
   * Null string entries will return null output string entries.
   * target Scalar should be string and should not be empty or null.
   *
   * @param target String to search for within each string.
   * @param replace Replacement string if target is found.
   * @return A new java column vector containing replaced strings
   */
  public ColumnVector stringReplace(Scalar target, Scalar replace) {
    return columnView.stringReplace(target, replace);
  }

  /**
   * For each string, replaces any character sequence matching the given pattern
   * using the replace template for back-references.
   *
   * Any null string entries return corresponding null output column entries.
   *
   * @param pattern The regular expression patterns to search within each string.
   * @param replace The replacement template for creating the output string.
   * @return A new java column vector containing the string results.
   */
  public ColumnVector stringReplaceWithBackrefs(String pattern, String replace) {
    return columnView.stringReplaceWithBackrefs(pattern, replace);
  }

  /**
   * Add '0' as padding to the left of each string.
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width The minimum number of characters for each string.
   * @return New column of strings.
   */
  public ColumnVector zfill(int width) {
    return columnView.zfill(width);
  }

  /**
   * Pad the Strings column until it reaches the desired length with spaces " " on the right.
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width the minimum number of characters for each string.
   * @return the new strings column.
   */
  public ColumnVector pad(int width) {
    return columnView.pad(width);
  }

  /**
   * Pad the Strings column until it reaches the desired length with spaces " ".
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width the minimum number of characters for each string.
   * @param side where to add new characters.
   * @return the new strings column.
   */
  public ColumnVector pad(int width, PadSide side) {
    return columnView.pad(width, side);
  }

  /**
   * Pad the Strings column until it reaches the desired length.
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width the minimum number of characters for each string.
   * @param side where to add new characters.
   * @param fillChar a single character string that holds what should be added.
   * @return the new strings column.
   */
  public ColumnVector pad(int width, PadSide side, String fillChar) {
    return columnView.pad(width, side, fillChar);
  }

  /**
   * Checks if each string in a column starts with a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param pattern scalar containing the string being searched for at the beginning of the column's strings.
   * @return A new java column vector containing the boolean results.
   */
  public ColumnVector startsWith(Scalar pattern) {
    return columnView.startsWith(pattern);
  }

  /**
   * Checks if each string in a column ends with a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param pattern scalar containing the string being searched for at the end of the column's strings.
   * @return A new java column vector containing the boolean results.
   */
  public ColumnVector endsWith(Scalar pattern) {
    return columnView.endsWith(pattern);
  }

  /**
   * Removes whitespace from the beginning and end of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public ColumnVector strip() {
   return columnView.strip();
  }

  /**
   * Removes the specified characters from the beginning and end of each string.
   * @param toStrip UTF-8 encoded characters to strip from each string.
   * @return A new java column vector containing the stripped strings.
   */
  public ColumnVector strip(Scalar toStrip) {
   return columnView.strip(toStrip);
  }

  /**
   * Removes whitespace from the beginning of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public ColumnVector lstrip() {
    return columnView.lstrip();
  }

  /**
   * Removes the specified characters from the beginning of each string.
   * @param toStrip UTF-8 encoded characters to strip from each string.
   * @return A new java column vector containing the stripped strings.
   */
  public ColumnVector lstrip(Scalar toStrip) {
    return columnView.lstrip(toStrip);
  }

  /**
   * Removes whitespace from the end of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public ColumnVector rstrip() {
    return columnView.rstrip();
  }

  /**
   * Removes the specified characters from the end of each string.
   * @param toStrip UTF-8 encoded characters to strip from each string.
   * @return A new java column vector containing the stripped strings.
   */
  public ColumnVector rstrip(Scalar toStrip) {
    return columnView.rstrip(toStrip);
  }

  /**
   * Checks if each string in a column contains a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param compString scalar containing the string being searched for.
   * @return A new java column vector containing the boolean results.
   */

  public ColumnVector stringContains(Scalar compString) {
    return columnView.stringContains(compString);
  }

  /**
   * Replaces values less than `lo` in `input` with `lo`,
   * and values greater than `hi` with `hi`.
   *
   * if `lo` is invalid, then lo will not be considered while
   * evaluating the input (Essentially considered minimum value of that type).
   * if `hi` is invalid, then hi will not be considered while
   * evaluating the input (Essentially considered maximum value of that type).
   *
   * ```
   * Example:
   * input: {1, 2, 3, NULL, 5, 6, 7}
   *
   * valid lo and hi
   * lo: 3, hi: 5, lo_replace : 0, hi_replace : 16
   * output:{0, 0, 3, NULL, 5, 16, 16}
   *
   * invalid lo
   * lo: NULL, hi: 5, lo_replace : 0, hi_replace : 16
   * output:{1, 2, 3, NULL, 5, 16, 16}
   *
   * invalid hi
   * lo: 3, hi: NULL, lo_replace : 0, hi_replace : 16
   * output:{0, 0, 3, NULL, 5, 6, 7}
   * ```
   * @param lo - Minimum clamp value. All elements less than `lo` will be replaced by `lo`.
   *           Ignored if null.
   * @param hi - Maximum clamp value. All elements greater than `hi` will be replaced by `hi`.
   *           Ignored if null.
   * @return Returns a new clamped column as per `lo` and `hi` boundaries
   */
  public ColumnVector clamp(Scalar lo, Scalar hi) {
    return columnView.clamp(lo, hi);
  }

  /**
   * Replaces values less than `lo` in `input` with `lo_replace`,
   * and values greater than `hi` with `hi_replace`.
   *
   * if `lo` is invalid, then lo will not be considered while
   * evaluating the input (Essentially considered minimum value of that type).
   * if `hi` is invalid, then hi will not be considered while
   * evaluating the input (Essentially considered maximum value of that type).
   *
   * @note: If `lo` is valid then `lo_replace` should be valid
   *        If `hi` is valid then `hi_replace` should be valid
   *
   * ```
   * Example:
   *    input: {1, 2, 3, NULL, 5, 6, 7}
   *
   *    valid lo and hi
   *    lo: 3, hi: 5, lo_replace : 0, hi_replace : 16
   *    output:{0, 0, 3, NULL, 5, 16, 16}
   *
   *    invalid lo
   *    lo: NULL, hi: 5, lo_replace : 0, hi_replace : 16
   *    output:{1, 2, 3, NULL, 5, 16, 16}
   *
   *    invalid hi
   *    lo: 3, hi: NULL, lo_replace : 0, hi_replace : 16
   *    output:{0, 0, 3, NULL, 5, 6, 7}
   * ```
   *
   * @param lo - Minimum clamp value. All elements less than `lo` will be replaced by `loReplace`. Ignored if null.
   * @param loReplace - All elements less than `lo` will be replaced by `loReplace`.
   * @param hi - Maximum clamp value. All elements greater than `hi` will be replaced by `hiReplace`. Ignored if null.
   * @param hiReplace - All elements greater than `hi` will be replaced by `hiReplace`.
   * @return - a new clamped column as per `lo` and `hi` boundaries
   */
  public ColumnVector clamp(Scalar lo, Scalar loReplace, Scalar hi, Scalar hiReplace) {
    return columnView.clamp(lo, loReplace, hi, hiReplace);
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
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   *
   * @param pattern Regex pattern to match to each string.
   * @return New ColumnVector of boolean results for each string.
   */
  public ColumnVector matchesRe(String pattern) {
    return columnView.matchesRe(pattern);
  }

  /**
   * Returns a boolean ColumnVector identifying rows which
   * match the given regex pattern starting at any location.
   *
   * ```
   * cv = ["abc","123","def456"]
   * result = cv.matches_re("\\d+")
   * r is now [false, true, true]
   * ```
   * Any null string entries return corresponding null output column entries.
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   *
   * @param pattern Regex pattern to match to each string.
   * @return New ColumnVector of boolean results for each string.
   */
  public ColumnVector containsRe(String pattern) {
    return columnView.containsRe(pattern);
  }

  /**
   * For each captured group specified in the given regular expression
   * return a column in the table. Null entries are added if the string
   * does not match. Any null inputs also result in null output entries.
   *
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   * @param pattern the pattern to use
   * @return the table of extracted matches
   * @throws CudfException if any error happens including if the RE does
   * not contain any capture groups.
   */
  public Table extractRe(String pattern) throws CudfException {
    return columnView.extractRe(pattern);
  }

  /** For a column of type List<Struct<String, String>> and a passed in String key, return a string column
   * for all the values in the struct that match the key, null otherwise.
   * @param key the String scalar to lookup in the column
   * @return a string column of values or nulls based on the lookup result
   */
  public ColumnVector getMapValue(Scalar key) {
    return columnView.getMapValue(key);
  }

  /////////////////////////////////////////////////////////////////////////////
  // INTERNAL/NATIVE ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Close all non-null buffers. Exceptions that occur during the process will
   * be aggregated into a single exception thrown at the end.
   */
  static void closeBuffers(AutoCloseable buffer) {
    Throwable toThrow = null;
    if (buffer != null) {
      try {
        buffer.close();
      } catch (Throwable t) {
        toThrow = t;
      }
    }
    if (toThrow != null) {
      throw new RuntimeException(toThrow);
    }
  }

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

  private static DeviceMemoryBufferView getData(long viewHandle) {
    return ColumnView.getDataBuffer(viewHandle);
  }

  private static DeviceMemoryBufferView getValid(long viewHandle) {
    return ColumnView.getValidityBuffer(viewHandle);
  }

  private static DeviceMemoryBufferView getOffsetsBuffer(long viewHandle) {
    return ColumnView.getOffsetsBuffer(viewHandle);
  }

  public ColumnView getChildColumnView(int childIndex) {
    if (!type.isNestedType()) {
      return null;
    }
    long childColumnView = ColumnView.getChildCvPointer(getNativeView(), childIndex);
    //this is returning a new ColumnView - must close this!
    return new ColumnView(childColumnView);
  }

  public BaseDeviceMemoryBuffer getData() {
    if (type.isNestedType()) {
      throw new IllegalStateException(" Lists and Structs at top level have no data");
    }
    return offHeap.getData();

  }

  public BaseDeviceMemoryBuffer getOffsets() {
    return offHeap.getOffsets();
  }

  public BaseDeviceMemoryBuffer getValid() {
    return offHeap.getValid();
  }

  public int getNumChildren() {
    if (!type.isNestedType()) {
      return 0;
    }
    return ColumnView.getNativeNumChildren(getNativeView());
  }

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  protected static final class OffHeapState extends MemoryCleaner.Cleaner {
    // This must be kept in sync with the native code
    public static final long UNKNOWN_NULL_COUNT = -1;
    private long columnHandle;
    private long viewHandle = 0;
    private List<MemoryBuffer> toClose = new ArrayList<>();


    /**
     * Make a column form an existing cudf::column *.
     */
    public OffHeapState(long columnHandle) {
      this.columnHandle = columnHandle;
      this.toClose.add(getData());
      this.toClose.add(getValid());
      this.toClose.add(getOffsets());
    }

    /**
     * Create a cudf::column_view from device side data.
     */
    public OffHeapState(DType type, int rows, Optional<Long> nullCount,
                        BaseDeviceMemoryBuffer data, BaseDeviceMemoryBuffer valid, BaseDeviceMemoryBuffer offsets,
                        List<BaseDeviceMemoryBuffer> buffers,
                        long[] childColumnViewHandles) {
      assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
          || !nullCount.isPresent();
      int nc = nullCount.orElse(UNKNOWN_NULL_COUNT).intValue();
      if (data != null) {
        this.toClose.add(data);
      }
      if (valid != null) {
        this.toClose.add(valid);
      }
      if (offsets != null) {
        this.toClose.add(offsets);
      }
      if (buffers != null) {
        toClose.addAll(buffers);
      }
      if (rows == 0 && !type.isNestedType()) {
        this.columnHandle = ColumnView.makeEmptyCudfColumn(type.typeId.getNativeId(), type.getScale());
      } else {
        long cd = data == null ? 0 : data.address;
        long cdSize = data == null ? 0 : data.length;
        long od = offsets == null ? 0 : offsets.address;
        long vd = valid == null ? 0 : valid.address;
        this.viewHandle = ColumnView.makeCudfColumnView(type.typeId.getNativeId(), type.getScale(),
            cd, cdSize, od, vd, nc, rows, childColumnViewHandles) ;
      }
    }

    /**
     * Create a cudf::column_view from contiguous device side data.
     */
    public OffHeapState(long viewHandle, DeviceMemoryBuffer contiguousBuffer) {
      assert viewHandle != 0;
      this.viewHandle = viewHandle;
      BaseDeviceMemoryBuffer valid = getValid();
      BaseDeviceMemoryBuffer data = getData();
      BaseDeviceMemoryBuffer offsets = getOffsets();
      toClose.add(data);
      toClose.add(valid);
      toClose.add(offsets);
      contiguousBuffer.incRefCount();
      toClose.add(contiguousBuffer);
    }

    public long getViewHandle() {
      if (viewHandle == 0) {
        viewHandle = ColumnView.getNativeColumnView(columnHandle);
      }
      return viewHandle;
    }

    public long getNativeRowCount() {
      return ColumnView.getNativeRowCount(getViewHandle());
    }

    public long getNativeNullCount() {
      if (viewHandle != 0) {
        return ColumnView.getNativeNullCount(getViewHandle());
      }
      return ColumnView.getNativeNullCountColumn(columnHandle);
    }

    private void setNativeNullCount(int nullCount) throws CudfException {
      assert viewHandle == 0 : "Cannot set the null count if a view has already been created";
      assert columnHandle != 0;
      ColumnView.setNativeNullCountColumn(columnHandle, nullCount);
    }

    public DType getNativeType() {
      return DType.fromNative(ColumnView.getNativeTypeId(getViewHandle()),
          ColumnView.getNativeTypeScale(getViewHandle()));
    }

    public int getNativeScale() {
      return ColumnView.getNativeTypeScale(getViewHandle());
    }

    public BaseDeviceMemoryBuffer getData() {
      return ColumnVector.getData(getViewHandle());
    }

    public BaseDeviceMemoryBuffer getValid() {
      return ColumnVector.getValid(getViewHandle());
    }

    public BaseDeviceMemoryBuffer getOffsets() {
      return getOffsetsBuffer(getViewHandle());
    }

    @Override
    public void noWarnLeakExpected() {
      super.noWarnLeakExpected();

      BaseDeviceMemoryBuffer valid = getValid();
      BaseDeviceMemoryBuffer data = getData();
      BaseDeviceMemoryBuffer offsets = getOffsets();
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
      return "(ID: " + id + " " + Long.toHexString(columnHandle == 0 ? viewHandle : columnHandle) + ")";
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long address = 0;

      // Always mark the resource as freed even if an exception is thrown.
      // We cannot know how far it progressed before the exception, and
      // therefore it is unsafe to retry.
      Throwable toThrow = null;
      if (viewHandle != 0) {
        address = viewHandle;
        try {
          ColumnView.deleteColumnView(viewHandle);
        } catch (Throwable t) {
          toThrow = t;
        } finally {
          viewHandle = 0;
        }
        neededCleanup = true;
      }
      if (columnHandle != 0) {
        if (address != 0) {
          address = columnHandle;
        }
        try {
          ColumnView.deleteCudfColumn(columnHandle);
        } catch (Throwable t) {
          if (toThrow != null) {
            toThrow.addSuppressed(t);
          } else {
            toThrow = t;
          }
        } finally {
          columnHandle = 0;
        }
        neededCleanup = true;
      }
      if (!toClose.isEmpty()) {
        try {
          for (MemoryBuffer toCloseBuff : toClose) {
            closeBuffers(toCloseBuff);
          }
        } catch (Throwable t) {
          if (toThrow != null) {
            toThrow.addSuppressed(t);
          } else {
            toThrow = t;
          }
        } finally {
          toClose.clear();
        }
        neededCleanup = true;
      }
      if (toThrow != null) {
        throw new RuntimeException(toThrow);
      }
      if (neededCleanup) {
        if (logErrorIfNotClean) {
          log.error("A DEVICE COLUMN VECTOR WAS LEAKED (ID: " + id + " " + Long.toHexString(address)+ ")");
          logRefCountDebug("Leaked vector");
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return viewHandle == 0 && columnHandle == 0 && toClose.isEmpty();
    }

    /**
     * This returns total memory allocated in device for the ColumnVector.
     * @return number of device bytes allocated for this column
     */
    public long getDeviceMemorySize() {
      BaseDeviceMemoryBuffer valid = getValid();
      BaseDeviceMemoryBuffer data = getData();
      BaseDeviceMemoryBuffer offsets = getOffsets();
      long size = valid != null ? valid.getLength() : 0;
      size += offsets != null ? offsets.getLength() : 0;
      size += data != null ? data.getLength() : 0;
      return size;
    }
  }

  public static ColumnVector createNestedColumnVector(DType type, int rows, HostMemoryBuffer data, HostMemoryBuffer valid, HostMemoryBuffer offsets,
                                                      Optional<Long> nullCount, List<HostColumnVectorCore> child) {
    return ColumnView.NestedColumnVector.createColumnVector(type, rows, data, valid, offsets, nullCount, child);
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
   * This method is evolving, unstable and currently test only.
   * Please use with caution and expect it to change in the future.
   */
  public static<T> ColumnVector fromLists(HostColumnVector.DataType dataType, List<T>... lists) {
    try (HostColumnVector host = HostColumnVector.fromLists(dataType, lists)) {
      return host.copyToDevice();
    }
  }

  /**
   * This method is evolving, unstable and currently test only.
   * Please use with caution and expect it to change in the future.
   */
  public static ColumnVector fromStructs(HostColumnVector.DataType dataType,
                                         List<HostColumnVector.StructData> lists) {
    try (HostColumnVector host = HostColumnVector.fromStructs(dataType, lists)) {
      return host.copyToDevice();
    }
  }

  /**
   * This method is evolving, unstable and currently test only.
   * Please use with caution and expect it to change in the future.
   */
  public static ColumnVector fromStructs(HostColumnVector.DataType dataType,
                                         HostColumnVector.StructData... lists) {
    try (HostColumnVector host = HostColumnVector.fromStructs(dataType, lists)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromBytes(byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned byte type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromUnsignedBytes(byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromShorts(short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned short type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromUnsignedShorts(short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromInts(int... values) {
    return build(DType.INT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned int type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromUnsignedInts(int... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromLongs(long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned long type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromUnsignedLongs(long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendArray(values));
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
  public static ColumnVector durationSecondsFromLongs(long... values) {
    return build(DType.DURATION_SECONDS, values.length, (b) -> b.appendArray(values));
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
  public static ColumnVector durationDaysFromInts(int... values) {
    return build(DType.DURATION_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector durationMilliSecondsFromLongs(long... values) {
    return build(DType.DURATION_MILLISECONDS, values.length, (b) -> b.appendArray(values));
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
  public static ColumnVector durationMicroSecondsFromLongs(long... values) {
    return build(DType.DURATION_MICROSECONDS, values.length, (b) -> b.appendArray(values));
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
  public static ColumnVector durationNanoSecondsFromLongs(long... values) {
    return build(DType.DURATION_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampNanoSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new decimal vector from unscaled values (int array) and scale.
   * The created vector is of type DType.DECIMAL32, whose max precision is 9.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static ColumnVector decimalFromInts(int scale, int... values) {
    try (HostColumnVector host = HostColumnVector.decimalFromInts(scale, values)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new decimal vector from unscaled values (long array) and scale.
   * The created vector is of type DType.DECIMAL64, whose max precision is 18.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static ColumnVector decimalFromLongs(int scale, long... values) {
    try (HostColumnVector host = HostColumnVector.decimalFromLongs(scale, values)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new decimal vector from double floats with specific DecimalType and RoundingMode.
   * All doubles will be rescaled if necessary, according to scale of input DecimalType and RoundingMode.
   * If any overflow occurs in extracting integral part, an IllegalArgumentException will be thrown.
   * This API is inefficient because of slow double -> decimal conversion, so it is mainly for testing.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static ColumnVector decimalFromDoubles(DType type, RoundingMode mode, double... values) {
    try (HostColumnVector host = HostColumnVector.decimalFromDoubles(type, mode, values)) {
      return host.copyToDevice();
    }
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
   * but is much slower than building from primitive array of unscaledValues.
   * Notice:
   *  1. All input BigDecimals should share same scale.
   *  2. The scale will be zero if all input values are null.
   */
  public static ColumnVector fromDecimals(BigDecimal... values) {
    try (HostColumnVector hcv = HostColumnVector.fromDecimals(values)) {
      return hcv.copyToDevice();
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
   * <p>
   * Java does not have an unsigned byte type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromBoxedUnsignedBytes(Byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendBoxed(values));
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
   * <p>
   * Java does not have an unsigned short type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromBoxedUnsignedShorts(Short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendBoxed(values));
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
   * <p>
   * Java does not have an unsigned int type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromBoxedUnsignedInts(Integer... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendBoxed(values));
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
   * <p>
   * Java does not have an unsigned long type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromBoxedUnsignedLongs(Long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendBoxed(values));
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
  public static ColumnVector durationDaysFromBoxedInts(Integer... values) {
    return build(DType.DURATION_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector durationSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_SECONDS, values.length, (b) -> b.appendBoxed(values));
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
  public static ColumnVector durationMilliSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
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
  public static ColumnVector durationMicroSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
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
  public static ColumnVector durationNanoSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
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
