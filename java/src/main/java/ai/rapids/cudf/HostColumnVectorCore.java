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

import static ai.rapids.cudf.HostColumnVector.OFFSET_SIZE;

/**
 * A class that holds Host side Column Vector APIs and the OffHeapState.
 * Any children of a HostColumnVector will be instantiated via this class.
 */
public class HostColumnVectorCore implements ColumnViewAccess<HostMemoryBuffer> {

  private static final Logger log = LoggerFactory.getLogger(HostColumnVector.class);

  protected final OffHeapState offHeap;
  protected final DType type;
  protected long rows;
  protected Optional<Long> nullCount;
  protected List<HostColumnVectorCore> children;


  public HostColumnVectorCore(DType type, long rows,
                              Optional<Long> nullCount, HostMemoryBuffer data, HostMemoryBuffer validity,
                              HostMemoryBuffer offsets, List<HostColumnVectorCore> nestedChildren) {
    this.offHeap = new OffHeapState(data, validity,  offsets);
    MemoryCleaner.register(this, offHeap);
    this.type = type;
    this.rows = rows;
    this.nullCount = nullCount;
    this.children = nestedChildren;
  }

  /**
   * Returns the type of this vector.
   */
  public DType getType() {
    return type;
  }

  /**
   * Returns the data buffer for a given host side column vector
   */
  HostMemoryBuffer getData() {
    return offHeap.data;
  }

  /**
   * Returns the validity buffer for a given host side column vector
   */
  HostMemoryBuffer getValidity() {
    return offHeap.valid;
  }

  /**
   * Returns the offset buffer
   */
  public HostMemoryBuffer getOffsets() {
    return offHeap.offsets;
  }

  @Override
  public long getColumnViewAddress() {
    throw new IllegalStateException("getColumnViewAddress is not supported on Host side");
  }

  @Override
  public ColumnViewAccess<HostMemoryBuffer> getChildColumnViewAccess(int childIndex) {
    return getNestedChildren().get(childIndex);
  }

  @Override
  public HostMemoryBuffer getDataBuffer() {
    return offHeap.data;
  }

  @Override
  public HostMemoryBuffer getOffsetBuffer() {
    return offHeap.offsets;
  }

  @Override
  public HostMemoryBuffer getValidityBuffer() {
    return offHeap.valid;
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
   * Returns the list of child host column vectors for a given host side column
   */
  List<HostColumnVectorCore> getNestedChildren() {
    return children;
  }

  /**
   * Get the data type of this column
   * @return DType of the column
   */
  @Override
  public DType getDataType() {
    return type;
  }

  /**
   * Returns the number of rows for a given host side column vector
   */
  @Override
  public long getRowCount() {
    return rows;
  }

  /**
   * Returns the number of rows for a given host side column vector, deprecated.
   */
  @Override
  @Deprecated
  public long getNumRows() {
    return rows;
  }

  /**
   * Returns the number of children for this column
   */
  @Override
  public int getNumChildren() {
    return children.size();
  }

  /**
   * Return the element at a given row for a give data type
   * @param rowIndex the row number
   * @return an object that would need to be casted to appropriate type based on this vector's data type
   */
  Object getElement(int rowIndex) {
    if (type == DType.LIST) {
      return getList(rowIndex);
    } else if (type == DType.STRUCT) {
      return getStruct(rowIndex);
    } else {
      if (isNull(rowIndex)) {
        return null;
      }
      return readValue(rowIndex);
    }
  }

  private Object getString(int rowIndex) {
    if (isNull(rowIndex)) {
      return null;
    }
    int start = (int)getStartListOffset(rowIndex);
    int end = (int)getEndListOffset(rowIndex);
    int size = end - start;
    byte[] rawData = new byte[size];
    if (size > 0) {
      offHeap.data.getBytes(rawData, 0, start, size);
      return new String(rawData);
    } else {
      return new String();
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  // DATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

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
    assert type.isBackedByByte() : type + " is not stored as a byte.";
    assertsForGet(index);
    return offHeap.data.getByte(index * type.getSizeInBytes());
  }

  /**
   * Get the value at index.
   */
  public final short getShort(long index) {
    assert type.isBackedByShort() : type + " is not stored as a short.";
    assertsForGet(index);
    return offHeap.data.getShort(index * type.getSizeInBytes());
  }

  /**
   * Get the value at index.
   */
  public final int getInt(long index) {
    assert type.isBackedByInt() : type + " is not stored as a int.";
    assertsForGet(index);
    return offHeap.data.getInt(index * type.getSizeInBytes());
  }

  /**
   * Get the starting byte offset for the string at index
   * Wraps getStartListOffset for backwards compatibility
   */
  long getStartStringOffset(long index) {
    return getStartListOffset(index);
  }

  /**
   * Get the starting element offset for the list or string at index
   */
  public long getStartListOffset(long index) {
    assert type.equals(DType.STRING) || type.equals(DType.LIST): type +
      " is not a supported string or list type.";
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    return offHeap.offsets.getInt(index * 4);
  }

  /**
   * Get the ending byte offset for the string at index.
   * Wraps getEndListOffset for backwards compatibility
   */
  long getEndStringOffset(long index) {
    return getEndListOffset(index);
  }

  /**
   * Get the ending element offset for the list or string at index.
   */
  public long getEndListOffset(long index) {
    assert type.equals(DType.STRING) || type.equals(DType.LIST): type +
      " is not a supported string or list type.";
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    // The offsets has one more entry than there are rows.
    return offHeap.offsets.getInt((index + 1) * 4);
  }

  /**
   * Get the value at index.
   */
  public final long getLong(long index) {
    // Timestamps with time values are stored as longs
    assert type.isBackedByLong(): type + " is not stored as a long.";
    assertsForGet(index);
    return offHeap.data.getLong(index * type.getSizeInBytes());
  }

  /**
   * Get the value at index.
   */
  public final float getFloat(long index) {
    assert type.equals(DType.FLOAT32) : type + " is not a supported float type.";
    assertsForGet(index);
    return offHeap.data.getFloat(index * type.getSizeInBytes());
  }

  /**
   * Get the value at index.
   */
  public final double getDouble(long index) {
    assert type.equals(DType.FLOAT64) : type + " is not a supported double type.";
    assertsForGet(index);
    return offHeap.data.getDouble(index * type.getSizeInBytes());
  }

  /**
   * Get the boolean value at index
   */
  public final boolean getBoolean(long index) {
    assert type.equals(DType.BOOL8) : type + " is not a supported boolean type.";
    assertsForGet(index);
    return offHeap.data.getBoolean(index * type.getSizeInBytes());
  }

  /**
   * Get the raw UTF8 bytes at index.  This API is faster than getJavaString, but still not
   * ideal because it is copying the data onto the heap.
   */
  public byte[] getUTF8(long index) {
    assert type.equals(DType.STRING) : type + " is not a supported string type.";
    assertsForGet(index);
    int start = (int)getStartListOffset(index);
    int size = (int)getEndListOffset(index) - start;
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

  /**
   * WARNING: Special case for lists of int8 or uint8, does not support null list values or lists
   *
   * Get array of bytes at index from a list column of int8 or uint8. The column may not be a list
   * of lists and may not have nulls.
   */
  public byte[] getBytesFromList(long rowIndex) {
    assert type.equals(DType.LIST) : type + " is not a supported list of bytes type.";
    HostColumnVectorCore listData = children.get(0);
    assert listData.type.equals(DType.INT8) || listData.type.equals(DType.UINT8)  : type +
      " is not a supported list of bytes type.";
    assert !listData.hasNulls() : "byte list column with nulls are not supported";
    assertsForGet(rowIndex);

    int start = (int)getStartListOffset(rowIndex);
    int end = (int)getEndListOffset(rowIndex);
    int size = end - start;

    byte[] result = new byte[size];
    listData.offHeap.data.getBytes(result, 0, start, size);
    return result;
  }

  /**
   * WARNING: Strictly for test only. This call is not efficient for production.
   */
  public List getList(long rowIndex) {
    assert rowIndex < rows;
    assert type == DType.LIST;
    List retList = new ArrayList();
    int start = (int)getStartListOffset(rowIndex);
    int end = (int)getEndListOffset(rowIndex);
    // check if null or empty
    if (isNull(rowIndex)) {
      return null;
    }
    for(int j = start; j < end; j++) {
      for (HostColumnVectorCore childHcv : children) {
        // lists have only 1 child
        retList.add(childHcv.getElement(j));
      }
    }
    return retList;
  }

  /**
   * WARNING: Strictly for test only. This call is not efficient for production.
   */
  public HostColumnVector.StructData getStruct(int rowIndex) {
    assert rowIndex < rows;
    assert type == DType.STRUCT;
    List<Object> retList = new ArrayList<>();
    // check if null or empty
    if (isNull(rowIndex)) {
      return null;
    }
    for (int k = 0; k < this.getNumChildren(); k++) {
      retList.add(children.get(k).getElement(rowIndex));
    }
    return new HostColumnVector.StructData(retList);
  }

  /**
   * Method that returns a boolean to indicate if the element at a given row index is null
   * @param rowIndex the row index
   * @return true if null else false
   */
  public boolean isNull(long rowIndex) {
    assert (rowIndex >= 0 && rowIndex < rows) : "index is out of range 0 <= " + rowIndex + " < " + rows;
    if (hasValidityVector()) {
      return BitVectorHelper.isNull(offHeap.valid, rowIndex);
    }
    return false;
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

  /**
   * Helper method that reads in a value at a given row index
   * @param rowIndex the row index
   * @return an object that would need to be casted to appropriate type based on this vector's data type
   */
  private Object readValue(int rowIndex) {
    assert rowIndex < rows;
    int rowOffset = rowIndex * type.getSizeInBytes();
    switch (type.typeId) {
      case INT32: // fall through
      case UINT32: // fall through
      case TIMESTAMP_DAYS:
      case DURATION_DAYS: return offHeap.data.getInt(rowOffset);
      case INT64: // fall through
      case UINT64: // fall through
      case DURATION_MICROSECONDS: // fall through
      case DURATION_MILLISECONDS: // fall through
      case DURATION_NANOSECONDS: // fall through
      case DURATION_SECONDS: // fall through
      case TIMESTAMP_MICROSECONDS: // fall through
      case TIMESTAMP_MILLISECONDS: // fall through
      case TIMESTAMP_NANOSECONDS: // fall through
      case TIMESTAMP_SECONDS: return offHeap.data.getLong(rowOffset);
      case FLOAT32: return offHeap.data.getFloat(rowOffset);
      case FLOAT64: return offHeap.data.getDouble(rowOffset);
      case UINT8: // fall through
      case INT8: return offHeap.data.getByte(rowOffset);
      case UINT16: // fall through
      case INT16: return offHeap.data.getShort(rowOffset);
      case BOOL8: return offHeap.data.getBoolean(rowOffset);
      case STRING: return getString(rowIndex);
      default: throw new UnsupportedOperationException("Do not support " + type);
    }
  }

  /**
   * Returns the amount of host memory used to store column/validity data (not metadata).
   */
  public long getHostMemorySize() {
    long totalSize = offHeap.getHostMemorySize();
    for (HostColumnVectorCore nhcv : children) {
      totalSize += nhcv.getHostMemorySize();
    }
    return totalSize;
  }

  /**
   * Close method for the column
   */
  @Override
  public void close() {
    for (HostColumnVectorCore child : children) {
      if (child != null) {
        child.close();
      }
    }
    offHeap.delRef();
    offHeap.cleanImpl(false);
  }

  @Override
  public String toString() {
    return "HostColumnVectorCore{" +
        "rows=" + rows +
        ", type=" + type +
        ", nullCount=" + nullCount +
        ", offHeap=" + offHeap +
        '}';
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
}
