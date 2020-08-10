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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Similar to a ColumnVector, but the data is stored in host memory and accessible directly from
 * the JVM. This class holds references to off heap memory and is reference counted to know when
 * to release it.  Call close to decrement the reference count when you are done with the column,
 * and call incRefCount to increment the reference count.
 */
public final class HostColumnVector implements AutoCloseable {
  /**
   * The size in bytes of an offset entry
   */
  static final int OFFSET_SIZE = DType.INT32.sizeInBytes;
  private static final Logger log = LoggerFactory.getLogger(HostColumnVector.class);

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  protected static class NestedHostColumnVector implements AutoCloseable {

    private HostMemoryBuffer data;
    private HostMemoryBuffer offsets;
    private HostMemoryBuffer validity;
    private DType type;
    private long rows;
    private Optional<Long> nullCount = Optional.empty();
    private List<NestedHostColumnVector> nestedChildren = new ArrayList<>();

    public NestedHostColumnVector(DType type, long rows,
                                  Optional<Long> nullCount, HostMemoryBuffer data, HostMemoryBuffer validity,
                                  HostMemoryBuffer offsets, List<NestedHostColumnVector> nestedChildren) {
      this.data = data;
      this.offsets = offsets;
      this.validity = validity;
      this.type = type;
      this.rows = rows;
      this.nullCount = nullCount;
      this.nestedChildren = nestedChildren;
    }

    @Override
    public void close() {
      for (NestedHostColumnVector child : nestedChildren) {
        if (child != null) {
          child.close();
        }
      }
      if (data != null) {
        data.close();
      }
      if (validity != null) {
        validity.close();
      }
      if (offsets != null) {
        offsets.close();
      }
    }

    HostMemoryBuffer getData() {
      return data;
    }

    HostMemoryBuffer getValidity() {
      return validity;
    }

    DType getType() {
      return type;
    }

    Optional<Long> getNullCount() {
      return nullCount;
    }

    List<NestedHostColumnVector> getNestedChildren() {
      return nestedChildren;
    }

    public HostMemoryBuffer getOffsets() {
      return offsets;
    }

    public long getRows() {
      return rows;
    }


    private Object getElement(int rowIndex) {
      if (type == DType.LIST) {
        List retList = new ArrayList();
        int start = offsets.getInt(rowIndex * DType.INT32.getSizeInBytes());
        int end = offsets.getInt((rowIndex + 1) * DType.INT32.getSizeInBytes());
        for (int j = start; j < end; j++) {
          retList.add(nestedChildren.get(0).getElement(j));
        }
        return retList;
      } else if (type == DType.STRING) {
        int start = offsets.getInt(rowIndex * DType.INT32.getSizeInBytes());
        int end = offsets.getInt((rowIndex + 1) * DType.INT32.getSizeInBytes());
        int size = end - start;
        byte[] rawData = new byte[size];
        if (size > 0) {
          data.getBytes(rawData, 0, start, size);
          return new String(rawData);
        } else if (isNull(rowIndex)) {
          return null;
        } else {
          return new String();
        }
      } else {
        if (isNull(rowIndex)) {
          return null;
        }
        int start = rowIndex * type.getSizeInBytes();
        return readValue(start);
      }
    }

    public boolean isNull(long index) {
      assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
      if (validity != null) {
        return BitVectorHelper.isNull(validity, index);
      }
      return false;
    }

    private Object readValue(int index){
      assert index < rows * type.getSizeInBytes();
      switch (type) {
        case INT32: return data.getInt(index);
        case INT64: return data.getLong(index);
        case FLOAT32: return data.getFloat(index);
        case FLOAT64: return data.getDouble(index);
        case INT8: return data.getByte(index);
        case INT16: return data.getShort(index);
        case BOOL8: return data.getBoolean(index);
        default: throw new UnsupportedOperationException("Do not support " + type);
      }
    }

    public long getHostMemorySize() {
      long totalSize = 0;
      if (data != null) {
        totalSize += data.length;
      }
      if (offsets != null) {
        totalSize += offsets.length;
      }
      if (data != null) {
        totalSize += validity.length;
      }
      for (NestedHostColumnVector nhcv : nestedChildren) {
        totalSize += nhcv.getHostMemorySize();
      }
      return totalSize;
    }
  }

  private final OffHeapState offHeap;
  private final DType type;
  private long rows;
  private Optional<Long> nullCount = Optional.empty();
  private int refCount;
  List<NestedHostColumnVector> children = new ArrayList<>();

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
   * @param nestedHcv          list of child NestedHostColumnVector(s) for complex types
   */

  //Constructor for lists
  HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
                   HostMemoryBuffer offsetBuffer, List<NestedHostColumnVector> nestedHcv) {
    this.offHeap = new OffHeapState(hostDataBuffer, hostValidityBuffer, offsetBuffer);
    MemoryCleaner.register(this, offHeap);
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;
    this.children = nestedHcv;
    refCount = 0;
    incRefCountInternal(true);
  }

  HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
                   HostMemoryBuffer offsetBuffer) {
    assert type != DType.LIST : "This constructor should not be used for list type";
    if (nullCount.isPresent() && nullCount.get() > 0 && hostValidityBuffer == null) {
      throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
    }
    if (type != DType.STRING && type != DType.LIST) {
      assert offsetBuffer == null : "offsets are only supported for STRING and LIST";
    }
    offHeap = new OffHeapState(hostDataBuffer, hostValidityBuffer, offsetBuffer);
    MemoryCleaner.register(this, offHeap);
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;

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
      for( NestedHostColumnVector child : children) {
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
   * Returns the number of rows in this vector.
   */
  public long getRowCount() {
    return rows;
  }

  /**
   * Returns the amount of host memory used to store column/validity data (not metadata).
   */
  public long getHostMemorySize() {
    long childTotalSize = 0;
    if (!children.isEmpty()) {
      for (NestedHostColumnVector nhcv : children) {
        childTotalSize += nhcv.getHostMemorySize();
      }
    }
    return offHeap.getHostMemorySize() + childTotalSize;
  }

  /**
   * Returns the type of this vector.
   */
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
      throw new IllegalStateException("Calculating an unknown null count on the host is not currently supported");
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
    return (offHeap.valid != null);
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
   * Copy the data to the device.
   */
  public ColumnVector copyToDevice() {
    if (rows == 0) {
      return new ColumnVector(type, 0, Optional.of(0L), null, null, null);
    }
    // The simplest way is just to copy the buffers and pass them down.
    DeviceMemoryBuffer data = null;
    DeviceMemoryBuffer valid = null;
    DeviceMemoryBuffer offsets = null;
    try {
      if (type != DType.LIST) {
        HostMemoryBuffer hdata = this.offHeap.data;
        if (hdata != null) {
          long dataLen = rows * type.sizeInBytes;
          if (type == DType.STRING) {
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
          long validLen = ColumnVector.getNativeValidPointerSize((int) rows);
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
        return ColumnVector.createNestedColumnVector(type, (int)rows, offHeap.valid, offHeap.offsets, nullCount, children.get(0));
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
  // DATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * WARNING: Strictly for test only. This call is not efficient for production.
   */
  List getList(long rowIndex) {
    assert rowIndex < rows;
    assert type == DType.LIST;
    List retList = new ArrayList();
    int start = offHeap.offsets.getInt(rowIndex * DType.INT32.getSizeInBytes());
    int end = offHeap.offsets.getInt((rowIndex + 1) * DType.INT32.getSizeInBytes());
    // check if null or empty
    if (start == end) {
      if (isNull(rowIndex)) {
        return null;
      }
    }
    for(int j = start; j < end; j++) {
      retList.add(children.get(0).getElement(j));
    }
    return retList;
  }

  /**
   * Check if the value at index is null or not.
   */
  public boolean isNull(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    if (hasValidityVector()) {
      return BitVectorHelper.isNull(offHeap.valid, index);
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
    if (hasValidityVector()) {
      if (nullCount.isPresent() && !hasNulls()) {
        return false;
      }
      return BitVectorHelper.isNull(offHeap.valid, index);
    }
    return false;
  }

  /**
   * Get access to the raw host buffer for this column.  This is intended to be used with a lot
   * of caution.  The lifetime of the buffer is tied to the lifetime of the column (Do not close
   * the buffer, as the column will take care of it).  Do not modify the contents of the buffer or
   * it might negatively impact what happens on the column.  The data must be on the host for this
   * to work.
   * @param type the type of buffer to get access to.
   * @return the underlying buffer or null if no buffer is associated with it for this column.
   * Please note that if the column is empty there may be no buffers at all associated with the
   * column.
   */
  public HostMemoryBuffer getHostBufferFor(BufferType type) {
    HostMemoryBuffer srcBuffer = null;
    switch(type) {
      case VALIDITY:
        srcBuffer = offHeap.valid;
        break;
      case OFFSET:
        srcBuffer = offHeap.offsets;
        break;
      case DATA:
        srcBuffer = offHeap.data;
        break;
      default:
        throw new IllegalArgumentException(type + " is not a supported buffer type.");
    }
    return srcBuffer;
  }

  void copyHostBufferBytes(byte[] dst, int dstOffset, BufferType src, long srcOffset,
                           int length) {
    assert dstOffset >= 0;
    assert srcOffset >= 0;
    assert length >= 0;
    assert dstOffset + length <= dst.length;

    HostMemoryBuffer srcBuffer = getHostBufferFor(src);

    assert srcOffset + length <= srcBuffer.length : "would copy off end of buffer "
        + srcOffset + " + " + length + " > " + srcBuffer.length;
    UnsafeMemoryAccessor.getBytes(dst, dstOffset,
        srcBuffer.getAddress() + srcOffset, length);
  }

  /**
   * Generic type independent asserts when getting a value from a single index.
   * @param index where to get the data from.
   */
  private void assertsForGet(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert !isNull(index) : " value at " + index + " is null";
  }

  /**
   * Get the value at index.
   */
  public byte getByte(long index) {
    assert type == DType.INT8 || type == DType.UINT8 || type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.data.getByte(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final short getShort(long index) {
    assert type == DType.INT16 || type == DType.UINT16;
    assertsForGet(index);
    return offHeap.data.getShort(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final int getInt(long index) {
    assert type.isBackedByInt();
    assertsForGet(index);
    return offHeap.data.getInt(index * type.sizeInBytes);
  }

  /**
   * Get the starting byte offset for the string at index
   */
  long getStartStringOffset(long index) {
    assert type == DType.STRING;
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    return offHeap.offsets.getInt(index * 4);
  }

  /**
   * Get the ending byte offset for the string at index.
   */
  long getEndStringOffset(long index) {
    assert type == DType.STRING || type == DType.LIST;
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    // The offsets has one more entry than there are rows.
    return offHeap.offsets.getInt((index + 1) * 4);
  }

  /**
   * Get the value at index.
   */
  public final long getLong(long index) {
    // Timestamps with time values are stored as longs
    assert type.isBackedByLong();
    assertsForGet(index);
    return offHeap.data.getLong(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final float getFloat(long index) {
    assert type == DType.FLOAT32;
    assertsForGet(index);
    return offHeap.data.getFloat(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final double getDouble(long index) {
    assert type == DType.FLOAT64;
    assertsForGet(index);
    return offHeap.data.getDouble(index * type.sizeInBytes);
  }

  /**
   * Get the boolean value at index
   */
  public final boolean getBoolean(long index) {
    assert type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.data.getBoolean(index * type.sizeInBytes);
  }

  /**
   * Get the raw UTF8 bytes at index.  This API is faster than getJavaString, but still not
   * ideal because it is copying the data onto the heap.
   */
  public byte[] getUTF8(long index) {
    assert type == DType.STRING;
    assertsForGet(index);
    int start = offHeap.offsets.getInt(index * OFFSET_SIZE);
    int size = offHeap.offsets.getInt((index + 1) * OFFSET_SIZE) - start;
    byte[] rawData = new byte[size];
    if (size > 0) {
      offHeap.data.getBytes(rawData, 0, start, size);
    }
    return rawData;
  }

  /**
   * Get the value at index.  This API is slow as it has to translate the
   * string representation.  Please use it with caution.
   */
  public String getJavaString(long index) {
    byte[] rawData = getUTF8(index);
    return new String(rawData, StandardCharsets.UTF_8);
  }

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  protected static final class OffHeapState extends MemoryCleaner.Cleaner {
    public HostMemoryBuffer data;
    public HostMemoryBuffer valid = null;
    public HostMemoryBuffer offsets = null;

    OffHeapState(HostMemoryBuffer data, HostMemoryBuffer valid, HostMemoryBuffer offsets) {
      this.data = data;
      this.valid = valid;
      this.offsets = offsets;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (data != null || valid != null || offsets != null) {
        try {
          ColumnVector.closeBuffers(data);
          ColumnVector.closeBuffers(offsets);
          ColumnVector.closeBuffers(valid);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          data = null;
          valid = null;
          offsets = null;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A HOST COLUMN VECTOR WAS LEAKED (ID: " + id + ")");
        logRefCountDebug("Leaked vector");
      }
      return neededCleanup;
    }

    @Override
    public void noWarnLeakExpected() {
      super.noWarnLeakExpected();
      if (data != null) {
        data.noWarnLeakExpected();
      }
      if (valid != null) {
        valid.noWarnLeakExpected();
      }
      if (offsets != null) {
        offsets.noWarnLeakExpected();
      }
    }

    @Override
    public boolean isClean() {
      return data == null && valid == null && offsets == null;
    }

    /**
     * This returns total memory allocated on the host for the ColumnVector.
     */
    public long getHostMemorySize() {
      long total = 0;
      if (valid != null) {
        total += valid.length;
      }
      if (data != null) {
        total += data.length;
      }
      if (offsets != null) {
        total += offsets.length;
      }
      return total;
    }

    @Override
    public String toString() {
      return "(ID: " + id + ")";
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

  public static<T> HostColumnVector fromLists(ColumnBuilder.DataType dataType, List<T>... values) {
    ColumnBuilder cb = new ColumnBuilder(dataType);
    cb.appendLists(values);
    return cb.build();
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
   * WARNING: Debug only method to print a passed in buffer
   */
  private static void printBuffer(HostMemoryBuffer buffer) {
    if (buffer == null) {
      return;
    }
    byte[] offsetbytes = new byte[(int)buffer.length];
    System.out.println("BUFFER length =" + offsetbytes.length);
    buffer.getBytes(offsetbytes, 0, 0, buffer.length);
    for (int i = 0; i < offsetbytes.length; i++) {
      System.out.print(offsetbytes[i]);
      if (i%4 == 0) {
        System.out.print(" ");
      }
    }
    System.out.println();
  }

  /**
   * Build
   */

  public static final class ColumnBuilder implements  AutoCloseable {

    public static final int INIT_OFFSET_SIZE = 10;
    private DType type;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private HostMemoryBuffer offsets;
    private long nullCount = 0l;
    //TODO nullable currently not used
    private boolean nullable;
    private long rows;
    private boolean built = false;
    private List<ColumnBuilder> childBuilders = new ArrayList<>();

    private int currentIndex = 0;
    private int currentByteIndex = 0;


    public ColumnBuilder(DataType type) {
      this.type = type.getType();
      this.nullable = type.isNullable();
      this.rows = type.getSize();
      for (int i = 0; i < type.getNumChildren(); i++) {
        childBuilders.add(new ColumnBuilder(type.getChild(i)));
      }
    }

    public HostColumnVector build() {
      List<NestedHostColumnVector> nestedHostColumnVectorList = new ArrayList<>();
      for (ColumnBuilder childBuilder : childBuilders) {
        nestedHostColumnVectorList.add(childBuilder.buildNestedInternal());
      }
      HostColumnVector hostColumnVector = new HostColumnVector(type, rows, Optional.of(nullCount), data, valid, offsets,
          nestedHostColumnVectorList);
      return hostColumnVector;
    }

    private NestedHostColumnVector buildNestedInternal() {
      List<NestedHostColumnVector> nestedHostColumnVectorList = new ArrayList<>();
      for (ColumnBuilder childBuilder : childBuilders) {
        nestedHostColumnVectorList.add(childBuilder.buildNestedInternal());
      }
      NestedHostColumnVector ret = new NestedHostColumnVector(type, rows, Optional.of(nullCount), data, valid, offsets, nestedHostColumnVectorList);
      return ret;
    }

    private void allocateBitmaskAndSetDefaultValues() {
      long bitmaskSize = ColumnVector.getNativeValidPointerSize((int) rows);
      valid = HostMemoryBuffer.allocate(bitmaskSize);
      valid.setMemory(0, bitmaskSize, (byte) 0xFF);
    }

    public ColumnBuilder appendLists(List... inputLists) {
      for (List inputList : inputLists) {
        // one row
        append(inputList);
      }
      return this;
    }

    private void initAndResizeOffsetBuffer() {
      if (this.offsets == null) {
        offsets = HostMemoryBuffer.allocate(INIT_OFFSET_SIZE);
        offsets.setInt(0, 0);
      } else {
        if (offsets.length <= currentIndex * OFFSET_SIZE + OFFSET_SIZE) {
          HostMemoryBuffer newOffset = HostMemoryBuffer.allocate(offsets.length * 2);
          try {
            newOffset.copyFromHostBuffer(0, offsets, 0, offsets.length);
            offsets.close();
            offsets = newOffset;
            newOffset = null;
          } finally {
            if (newOffset != null) {
              newOffset.close();
            }
          }
        }
      }
    }

    private void initValidBuffer() {
      if (this.valid == null) {
        allocateBitmaskAndSetDefaultValues();
      }
    }

    private void setNullAt(int index) {
      initValidBuffer();
      nullCount += BitVectorHelper.setNullAt(valid, index);
    }

    public final ColumnBuilder appendNull() {
      setNullAt(currentIndex);
      currentIndex++;
      if (type == DType.STRING) {
        initAndResizeOffsetBuffer();
        offsets.setInt(currentIndex * OFFSET_SIZE, currentByteIndex);
      }
      return this;
    }

    private ColumnBuilder append(List inputList) {
      assert type == DType.LIST;
      // We know lists have only 1 child
      ColumnBuilder childBuilder = childBuilders.get(0);
      if (inputList == null) {
        initValidBuffer();
        setNullAt(currentIndex);
      } else {
        for (Object listElement : inputList) {
          if (listElement == null) {
            childBuilder.appendNull();
          } else if (listElement instanceof Integer) {
            childBuilder.append((Integer) listElement);
          } else if (listElement instanceof String) {
            childBuilder.append((String) listElement);
          }  else if (listElement instanceof Double) {
            childBuilder.append((Double) listElement);
          } else if (listElement instanceof List) {
            childBuilder.append((List) listElement);
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
          }
        }
      }
      currentIndex++;
      initAndResizeOffsetBuffer();
      offsets.setInt(currentIndex * OFFSET_SIZE, childBuilder.getCurrentIndex());
      return this;
    }

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
      assert type == DType.INT8;
      assert currentIndex < rows;
      resizeDataBuffer(currentIndex * type.sizeInBytes + type.getSizeInBytes());
      data.setByte(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      currentByteIndex += type.sizeInBytes;
      return this;
    }

    public final ColumnBuilder append(short value) {
      assert type == DType.INT16;
      assert currentIndex < rows;
      resizeDataBuffer(currentIndex * type.sizeInBytes + type.getSizeInBytes());
      data.setShort(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      currentByteIndex += type.sizeInBytes;
      return this;
    }

    public final ColumnBuilder append(int value) {
      assert type.isBackedByInt();
      assert currentIndex < rows;
      resizeDataBuffer(currentIndex * type.sizeInBytes + type.getSizeInBytes());
      data.setInt(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      currentByteIndex += type.sizeInBytes;
      return this;
    }

    public final ColumnBuilder append(long value) {
      assert type == DType.INT64;
      assert currentIndex < rows;
      resizeDataBuffer(currentIndex * type.sizeInBytes + type.getSizeInBytes());
      data.setLong(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      currentByteIndex += type.sizeInBytes;
      return this;
    }

    public final ColumnBuilder append(float value) {
      assert type == DType.FLOAT32;
      assert currentIndex < rows;
      resizeDataBuffer(currentIndex * type.sizeInBytes + type.getSizeInBytes());
      data.setFloat(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      currentByteIndex += type.sizeInBytes;
      return this;
    }

    public final ColumnBuilder append(double value) {
      assert type == DType.FLOAT64;
      assert currentIndex < rows;
      resizeDataBuffer(currentIndex * type.sizeInBytes + type.getSizeInBytes());
      data.setDouble(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      currentByteIndex += type.sizeInBytes;
      return this;
    }

    public final ColumnBuilder append(boolean value) {
      assert type == DType.BOOL8;
      assert currentIndex < rows;
      resizeDataBuffer(currentIndex * type.sizeInBytes + type.getSizeInBytes());
      data.setBoolean(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      currentByteIndex += type.sizeInBytes;
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
      assert type == DType.STRING : " type " + type + " is not String";
      assert currentIndex < rows + 1;
      resizeDataBuffer(length);
      if (length > 0) {
        data.setBytes(currentByteIndex, value, srcOffset, length);
      }
      currentByteIndex += length;
      currentIndex++;
      initAndResizeOffsetBuffer();
      offsets.setInt(currentIndex * OFFSET_SIZE, currentByteIndex);
      return this;
    }

    private void resizeDataBuffer(int length) {
      // just for strings we want to throw a real exception if we would overrun the buffer
      if (data == null) {
        data = HostMemoryBuffer.allocate(length);
        return;
      }
      long oldLen = data.getLength();
      long newLen = oldLen;
      while (currentByteIndex + length > newLen) {
        newLen *= 2;
      }
      if (newLen > Integer.MAX_VALUE) {
        throw new IllegalStateException("A string buffer is not supported over 2GB in size");
      }
      if (newLen != oldLen) {
        // need to grow the size of the buffer.
        HostMemoryBuffer newData = HostMemoryBuffer.allocate(newLen);
        try {
          newData.copyFromHostBuffer(0, data, 0, currentByteIndex);
          data.close();
          data = newData;
          newData = null;
        } finally {
          if (newData != null) {
            newData.close();
          }
        }
      }
    }

    public ColumnBuilder getChild(int index) {
      return childBuilders.get(index);
    }

    @Override
    public void close() throws Exception {
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

    public static abstract class DataType {
      abstract DType getType();
      abstract boolean isNullable();
      abstract long getSize();
      abstract DataType getChild(int index);
      abstract int getNumChildren();
    }

    public static class ListType extends DataType {
      private boolean isNullable;
      private long size;
      private DataType child;

      public ListType(boolean isNullable, long size, DataType child) {
        this.isNullable = isNullable;
        this.size = size;
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
      long getSize() {
        return size;
      }

      @Override
      DataType getChild(int index) {
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

    public static class BasicType extends DataType {
      private DType type;
      private boolean isNullable;
      private long size;

      public BasicType(boolean isNullable, long size, DType type) {
        this.isNullable = isNullable;
        this.size = size;
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
      long getSize() {
        return size;
      }

      @Override
      DataType getChild(int index) {
        return null;
      }

      @Override
      int getNumChildren() {
        return 0;
      }
    }
  }

  public static final class Builder implements AutoCloseable {
    public static final int INIT_OFFSET_SIZE = 10;
    private final long rows;
    private final DType type;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private HostMemoryBuffer offsets;
    private long currentIndex = 0;
    private long currentListIndex = 0;
    private long nullCount;
    private int currentStringByteIndex = 0;
    private int currentListCount = 0;
    private boolean built;
    private boolean needToAdd = false;

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
      if (type == DType.STRING) {
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
        this.data = HostMemoryBuffer.allocate(rows * type.sizeInBytes);
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
      assert type == DType.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value ? (byte)1 : (byte)0);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value) {
      assert type == DType.INT8 || type == DType.UINT8 || type == DType.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value, long count) {
      assert (count + currentIndex) <= rows;
      assert type == DType.INT8 || type == DType.UINT8 || type == DType.BOOL8;
      data.setMemory(currentIndex * type.sizeInBytes, count, value);
      currentIndex += count;
      return this;
    }

    public final Builder append(short value) {
      assert type == DType.INT16 || type == DType.UINT16;
      assert currentIndex < rows;
      data.setShort(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(int value) {
      assert type.isBackedByInt();
      assert currentIndex < rows;
      data.setInt(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(long value) {
      assert type.isBackedByLong();
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
      assert type == DType.STRING;
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
      assert type == DType.INT8 || type == DType.UINT8 || type == DType.BOOL8;
      data.setBytes(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(short... values) {
      assert type == DType.INT16 || type == DType.UINT16;
      assert (values.length + currentIndex) <= rows;
      data.setShorts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(int... values) {
      assert type.isBackedByInt();
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(long... values) {
      assert type.isBackedByLong();
      assert (values.length + currentIndex) <= rows;
      data.setLongs(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(float... values) {
      assert type == DType.FLOAT32;
      assert (values.length + currentIndex) <= rows;
      data.setFloats(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(double... values) {
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
      assert columnVector.type == type;

      if (type == DType.STRING) {
        throw new UnsupportedOperationException(
            "Appending a string column vector client side is not currently supported");
      } else {
        data.copyFromHostBuffer(currentIndex * type.sizeInBytes, columnVector.offHeap.data,
            0L,
            columnVector.getRowCount() * type.sizeInBytes);
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
      long bitmaskSize = ColumnVector.getNativeValidPointerSize((int) rows);
      valid = HostMemoryBuffer.allocate(bitmaskSize);
      valid.setMemory(0, bitmaskSize, (byte) 0xFF);
    }

    /**
     * Append null value.
     */
    public final Builder appendNull() {
      setNullAt(currentIndex);
      currentIndex++;
      if (type == DType.STRING) {
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
}
