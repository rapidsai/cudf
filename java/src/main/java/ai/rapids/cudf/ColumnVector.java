/*
 *
 *  Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
public final class ColumnVector extends ColumnView {
  private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final OffHeapState offHeap;
  private Optional<Long> nullCount = Optional.empty();
  private int refCount;

  /**
   * Wrap an existing on device cudf::column with the corresponding ColumnVector. The new
   * ColumnVector takes ownership of the pointer and will free it when the ref count reaches zero.
   * @param nativePointer host address of the cudf::column object which will be
   *                      owned by this instance.
   */
  public ColumnVector(long nativePointer) {
    super(getColumnViewFromColumn(nativePointer));
    assert nativePointer != 0;
    offHeap = new OffHeapState(nativePointer);
    MemoryCleaner.register(this, offHeap);
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
    super(ColumnVector.initViewHandle(
        type, (int)rows, nullCount.orElse(UNKNOWN_NULL_COUNT).intValue(),
        dataBuffer, validityBuffer, offsetBuffer, null));
    assert !type.equals(DType.LIST) : "This constructor should not be used for list type";
    if (!type.equals(DType.STRING)) {
      assert offsetBuffer == null : "offsets are only supported for STRING";
    }
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
    offHeap = new OffHeapState(type, (int) rows, dataBuffer, validityBuffer,
        offsetBuffer, null, viewHandle);
    MemoryCleaner.register(this, offHeap);
    this.nullCount = nullCount;

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
   * @param toClose  List of buffers to track adn close once done, usually in case of children
   * @param childHandles array of longs for child column view handles.
   */
  public ColumnVector(DType type, long rows, Optional<Long> nullCount,
                      DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer,
                      DeviceMemoryBuffer offsetBuffer, List<DeviceMemoryBuffer> toClose, long[] childHandles) {
    super(initViewHandle(type, (int)rows, nullCount.orElse(UNKNOWN_NULL_COUNT).intValue(),
        dataBuffer, validityBuffer,
        offsetBuffer, childHandles));
    if (!type.equals(DType.STRING) && !type.equals(DType.LIST)) {
      assert offsetBuffer == null : "offsets are only supported for STRING, LISTS";
    }
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
    offHeap = new OffHeapState(type, (int) rows, dataBuffer, validityBuffer, offsetBuffer,
            toClose, viewHandle);
    MemoryCleaner.register(this, offHeap);

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
    super(viewAddress);
    offHeap = new OffHeapState(viewAddress, contiguousBuffer);
    MemoryCleaner.register(this, offHeap);
    // TODO we may want to ask for the null count anyways...
    this.nullCount = Optional.empty();

    this.refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * Retrieves the column_view for a cudf::column and if it fails to do so, the column is deleted
   * and the exception is thrown to the caller.
   * @param nativePointer the cudf::column handle
   * @return the column_view handle
   */
  private static long getColumnViewFromColumn(long nativePointer) {
    try {
      return ColumnVector.getNativeColumnView(nativePointer);
    } catch (CudfException ce) {
      deleteCudfColumn(nativePointer);
      throw ce;
    }
  }


  private static long initViewHandle(DType type, int rows, int nc, DeviceMemoryBuffer dataBuffer,
                                     DeviceMemoryBuffer validityBuffer,
                                     DeviceMemoryBuffer offsetBuffer, long[] childHandles) {
    long cd = dataBuffer == null ? 0 : dataBuffer.address;
    long cdSize = dataBuffer == null ? 0 : dataBuffer.length;
    long od = offsetBuffer == null ? 0 : offsetBuffer.address;
    long vd = validityBuffer == null ? 0 : validityBuffer.address;
    return makeCudfColumnView(type.typeId.getNativeId(), type.getScale(), cd, cdSize,
        od, vd, nc, rows, childHandles) ;
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

  /**
   * Create a new vector of length rows, where each row is filled with the Scalar's
   * value
   * @param scalar - Scalar to use to fill rows
   * @param rows - Number of rows in the new ColumnVector
   * @return - new ColumnVector
   */
  public static ColumnVector fromScalar(Scalar scalar, int rows) {
    long columnHandle = fromScalar(scalar.getScalarHandle(), rows);
    return new ColumnVector(columnHandle);
  }

  /**
   * Create a new struct vector made up of existing columns. Note that this will copy
   * the contents of the input columns to make a new vector. If you only want to
   * do a quick temporary computation you can use ColumnView.makeStructView.
   * @param columns the columns to make the struct from.
   * @return the new ColumnVector
   */
  public static ColumnVector makeStruct(ColumnView... columns) {
    try (ColumnView cv = ColumnView.makeStructView(columns)) {
      return cv.copyToColumnVector();
    }
  }

  /**
   * Create a new struct vector made up of existing columns. Note that this will copy
   * the contents of the input columns to make a new vector. If you only want to
   * do a quick temporary computation you can use ColumnView.makeStructView.
   * @param rows the number of rows in the struct. Used for structs with no children.
   * @param columns the columns to make the struct from.
   * @return the new ColumnVector
   */
  public static ColumnVector makeStruct(long rows, ColumnView... columns) {
    try (ColumnView cv = ColumnView.makeStructView(rows, columns)) {
      return cv.copyToColumnVector();
    }
  }

  /**
   * Create a LIST column from the given columns. Each list in the returned column will have the
   * same number of entries in it as columns passed into this method. Be careful about the
   * number of rows passed in as there are limits on the maximum output size supported for
   * column lists.
   * @param columns the columns to make up the list column, in the order they will appear in the
   *                resulting lists.
   * @return the new LIST ColumnVector
   */
  public static ColumnVector makeList(ColumnView... columns) {
    if (columns.length <= 0) {
      throw new IllegalArgumentException("At least one column is needed to get the row count");
    }
    return makeList(columns[0].getRowCount(), columns[0].getType(), columns);
  }

  /**
   * Create a LIST column from the given columns. Each list in the returned column will have the
   * same number of entries in it as columns passed into this method. Be careful about the
   * number of rows passed in as there are limits on the maximum output size supported for
   * column lists.
   * @param rows the number of rows to create, for the special case of an empty list.
   * @param type the type of the child column, for the special case of an empty list.
   * @param columns the columns to make up the list column, in the order they will appear in the
   *                resulting lists.
   * @return the new LIST ColumnVector
   */
  public static ColumnVector makeList(long rows, DType type, ColumnView... columns) {
    long[] handles = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      ColumnView cv = columns[i];
      if (rows != cv.getRowCount()) {
        throw new IllegalArgumentException("All columns must have the same number of rows");
      }
      if (!type.equals(cv.getType())) {
        throw new IllegalArgumentException("All columns must have the same type");
      }

      handles[i] = cv.getNativeView();
    }
    if (columns.length == 0 && type.isNestedType()) {
      throw new IllegalArgumentException(
          "Creating an empty list column of nested types is not currently supported");
    }
    return new ColumnVector(makeList(handles, type.typeId.nativeId, type.getScale(), rows));
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
    if (!initialValue.isValid() || !step.isValid()) {
      throw new IllegalArgumentException("nulls are not supported in sequence");
    }
    return new ColumnVector(sequence(initialValue.getScalarHandle(), step.getScalarHandle(), rows));
  }

  /**
   * Create a new vector of length rows, starting at the initialValue and going by 1 each time.
   * Only numeric types are supported.
   * @param initialValue the initial value to start at.
   * @param rows the total number of rows
   * @return the new ColumnVector.
   */
  public static ColumnVector sequence(Scalar initialValue, int rows) {
    if (!initialValue.isValid()) {
      throw new IllegalArgumentException("nulls are not supported in sequence");
    }
    return new ColumnVector(sequence(initialValue.getScalarHandle(), 0, rows));
  }
  /**
   * Create a new vector by concatenating multiple columns together.
   * Note that all columns must have the same type.
   */
  public static ColumnVector concatenate(ColumnView... columns) {
    if (columns.length < 2) {
      throw new IllegalArgumentException("Concatenate requires 2 or more columns");
    }
    long[] columnHandles = new long[columns.length];
    for (int i = 0; i < columns.length; ++i) {
      columnHandles[i] = columns[i].getNativeView();
    }
    return new ColumnVector(concatenate(columnHandles));
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column
   * into a single string row of a new column with no separator string inserted between each
   * combined string and maintaining null values in combined rows.
   * @param columns array of columns containing strings.
   * @return A new java column vector containing the concatenated strings.
   */
  public static ColumnVector stringConcatenate(ColumnView[] columns) {
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
   * @param columns array of columns containing strings, must be more than 2 columns
   * @return A new java column vector containing the concatenated strings.
   */
  public static ColumnVector stringConcatenate(Scalar separator, Scalar narep, ColumnView[] columns) {
    assert columns.length >= 2 : ".stringConcatenate() operation requires at least 2 columns";
    assert separator != null : "separator scalar provided may not be null";
    assert separator.getType().equals(DType.STRING) : "separator scalar must be a string scalar";
    assert narep != null : "narep scalar provided may not be null";
    assert narep.getType().equals(DType.STRING) : "narep scalar must be a string scalar";
    long size = columns[0].getRowCount();
    long[] column_views = new long[columns.length];

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getType().equals(DType.STRING) : "All columns must be of type string for .cat() operation";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must have the same number of rows";
      column_views[i] = columns[i].getNativeView();
    }

    return new ColumnVector(stringConcatenation(column_views, separator.getScalarHandle(), narep.getScalarHandle()));
  }

  /**
   * Create a new vector containing the MD5 hash of each row in the table.
   *
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32 character hex strings representing each row's hash value.
   */
  public static ColumnVector md5Hash(ColumnView... columns) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("MD5 hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];
    long size = columns[0].getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      assert !columns[i].getType().isDurationType() : "Unsupported column type Duration";
      assert !columns[i].getType().isTimestampType() : "Unsupported column type Timestamp";
      assert !columns[i].getType().isNestedType() || columns[i].getType().equals(DType.LIST) :
          "Unsupported nested type column";
      columnViews[i] = columns[i].getNativeView();
    }
    return new ColumnVector(hash(columnViews, HashType.HASH_MD5.getNativeId(), new int[0], 0));
  }

  /**
   * Create a new vector containing the murmur3 hash of each row in the table.
   *
   * @param seed integer seed for the murmur3 hash function
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32-bit values representing each row's hash value.
   */
  public static ColumnVector serial32BitMurmurHash3(int seed, ColumnView columns[]) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("Murmur3 hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];
    long size = columns[0].getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      assert !columns[i].getType().isDurationType() : "Unsupported column type Duration";
      assert !columns[i].getType().isTimestampType() : "Unsupported column type Timestamp";
      assert !columns[i].getType().isNestedType() : "Unsupported column of nested type";
      columnViews[i] = columns[i].getNativeView();
    }
    return new ColumnVector(hash(columnViews, HashType.HASH_SERIAL_MURMUR3.getNativeId(), new int[0], seed));
  }

  /**
   * Create a new vector containing the murmur3 hash of each row in the table, seed defaulted to 0.
   *
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32-bit values representing each row's hash value.
   */
  public static ColumnVector serial32BitMurmurHash3(ColumnView columns[]) {
    return serial32BitMurmurHash3(0, columns);
  }

  /**
   * Create a new vector containing spark's 32-bit murmur3 hash of each row in the table.
   * Spark's murmur3 hash uses a different tail processing algorithm.
   *
   * @param seed integer seed for the murmur3 hash function
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32-bit values representing each row's hash value.
   */
  public static ColumnVector spark32BitMurmurHash3(int seed, ColumnView columns[]) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("Murmur3 hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];
    long size = columns[0].getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      assert !columns[i].getType().isDurationType() : "Unsupported column type Duration";
      assert !columns[i].getType().isTimestampType() : "Unsupported column type Timestamp";
      assert !columns[i].getType().isNestedType() : "Unsupported column of nested type";
      columnViews[i] = columns[i].getNativeView();
    }
    return new ColumnVector(hash(columnViews, HashType.HASH_SPARK_MURMUR3.getNativeId(), new int[0], seed));
  }

  /**
   * Create a new vector containing spark's 32-bit murmur3 hash of each row in the table with the
   * seed set to 0. Spark's murmur3 hash uses a different tail processing algorithm.
   *
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32-bit values representing each row's hash value.
   */
  public static ColumnVector spark32BitMurmurHash3(ColumnView columns[]) {
    return spark32BitMurmurHash3(0, columns);
  }

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
  @Override
  public ColumnVector castTo(DType type) {
    if (this.type.equals(type)) {
      // Optimization
      return incRefCount();
    }
    return super.castTo(type);
  }

  /////////////////////////////////////////////////////////////////////////////
  // NATIVE METHODS
  /////////////////////////////////////////////////////////////////////////////

  private static native long sequence(long initialValue, long step, int rows);

  private static native long fromScalar(long scalarHandle, int rowCount) throws CudfException;

  private static native long makeList(long[] handles, long typeHandle, int scale, long rows)
      throws CudfException;

  private static native long concatenate(long[] viewHandles) throws CudfException;

  /**
   * Native method to hash each row of the given table. Hashing function dispatched on the
   * native side using the hashId.
   *
   * @param viewHandles array of native handles to the cudf::column_view columns being operated on.
   * @param hashId integer native ID of the hashing function identifier HashType.
   * @param initialValues array of integer values, one per column, only used by non-serial murmur3
   *                      hash. Each element's hash value is merged with its column's initial value
   *                      before the row is merged into a single value.
   * @param seed integer seed for the hash. Only used by serial murmur3 hash.
   * @return native handle of the resulting cudf column containing the hex-string hashing results.
   */
  private static native long hash(long[] viewHandles, int hashId, int[] initialValues,
                                  int seed) throws CudfException;

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
  static native long getNativeColumnView(long cudfColumnHandle) throws CudfException;

  static native long makeEmptyCudfColumn(int type, int scale);

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
    public OffHeapState(DType type, int rows,
                        DeviceMemoryBuffer data, DeviceMemoryBuffer valid, DeviceMemoryBuffer offsets,
                        List<DeviceMemoryBuffer> buffers,
                        long viewHandle) {
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
        this.columnHandle = makeEmptyCudfColumn(type.typeId.getNativeId(), type.getScale());
      } else {
        this.viewHandle = viewHandle;
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
        viewHandle = ColumnVector.getNativeColumnView(columnHandle);
      }
      return viewHandle;
    }

    public long getNativeNullCount() {
      if (viewHandle != 0) {
        return ColumnView.getNativeNullCount(getViewHandle());
      }
      return getNativeNullCountColumn(columnHandle);
    }

    private void setNativeNullCount(int nullCount) throws CudfException {
      assert viewHandle == 0 : "Cannot set the null count if a view has already been created";
      assert columnHandle != 0;
      setNativeNullCountColumn(columnHandle, nullCount);
    }

    public BaseDeviceMemoryBuffer getData() {
      return getDataBuffer(getViewHandle());
    }

    public BaseDeviceMemoryBuffer getValid() {
      return getValidityBuffer(getViewHandle());
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
          ColumnVector.deleteCudfColumn(columnHandle);
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
   * This method is evolving, unstable and currently test only.
   * Please use with caution and expect it to change in the future.
   */
  public static ColumnVector emptyStructs(HostColumnVector.DataType dataType, long numRows) {
    try (HostColumnVector host = HostColumnVector.emptyStructs(dataType, numRows)) {
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
