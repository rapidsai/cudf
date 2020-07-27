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

import java.io.*;
import java.util.ArrayList;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

import static java.nio.charset.StandardCharsets.UTF_8;


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

  private final OffHeapState offHeap;
  private final DType type;
  private List<DType> allTypes = new ArrayList<>();
  private long rows;
  private List<Long> allRows = new ArrayList<>();
  private Optional<Long> nullCount = Optional.empty();
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
   */
  HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                   HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
                   HostMemoryBuffer offsetBuffer) {
    if (nullCount.isPresent() && nullCount.get() > 0 && hostValidityBuffer == null) {
      throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
    }
    if (type != DType.STRING && type != DType.LIST) {
      assert offsetBuffer == null : "offsets are only supported for STRING";
    }
    List<HostMemoryBuffer> validsBuffs = new ArrayList<>();
    List<HostMemoryBuffer> offsetBuffs = new ArrayList<>();
    validsBuffs.add(hostValidityBuffer);
    offsetBuffs.add(offsetBuffer);
    offHeap = new OffHeapState(hostDataBuffer, validsBuffs, offsetBuffs);
    MemoryCleaner.register(this, offHeap);
    this.rows = rows;
    this.allRows.add(rows);
    this.nullCount = nullCount;
    this.type = type;
    this.allTypes.add(type);

    refCount = 0;
    incRefCountInternal(true);
  }

  //Constructor for lists
  HostColumnVector(List<DType> type, List<Long> rows, Optional<Long> nullCount,
                   HostMemoryBuffer hostDataBuffer, List<HostMemoryBuffer> hostValidityBuffer,
                   List<HostMemoryBuffer> offsetBuffer) {
    if (nullCount.isPresent() && nullCount.get() > 0 && hostValidityBuffer == null) {
      throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
    }
    if (type.get(0) != DType.LIST) {
      assert offsetBuffer == null : "offsets are only supported for STRING";
    }
    offHeap = new OffHeapState(hostDataBuffer, hostValidityBuffer, offsetBuffer);
    System.out.println("PRINT IN HCV" + hostDataBuffer);
    MemoryCleaner.register(this, offHeap);
    this.rows = rows.get(0);
    this.allRows = rows;
    this.nullCount = nullCount;
    this.type = type.get(0);
    this.allTypes = type;

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
    return offHeap.getHostMemorySize();
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
        HostMemoryBuffer hvalid = this.offHeap.valid.get(0);
        if (hvalid != null) {
          long validLen = ColumnVector.getNativeValidPointerSize((int) rows);
          valid = DeviceMemoryBuffer.allocate(validLen);
          valid.copyFromHostBuffer(hvalid, 0, validLen);
        }

        HostMemoryBuffer hoff = this.offHeap.offsets.get(0);
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
        int depth = allTypes.size() -1 ; // 0,1,2 for List<List<int>>
        int offsetLen = this.offHeap.offsets.size();
        int validityLen = this.offHeap.valid.size();
        int rowsLen = this.allRows.size();
        long prev = 0l;
        ColumnVector newCol = null;
        for (int i = depth; i >= 0; i--) {
          DType colType = allTypes.get(i);
          long colRows = allRows.get(i);
          //TODO: fix this so that only one level has data
          HostMemoryBuffer colData = (i == depth) ? offHeap.data : null;
          HostMemoryBuffer colValid = i < offHeap.valid.size() ? offHeap.valid.get(i) : null;
          HostMemoryBuffer colOffsets = i < offHeap.offsets.size() ? offHeap.offsets.get(i) : null;
          newCol = createColumnVector(colType, colRows, nullCount, colData, colValid,
              colOffsets, prev);
          prev = newCol.getNativeView();
        }
      return newCol;
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

  ColumnVector createColumnVector(DType type, long rows, Optional<Long> nullCount,
                                  HostMemoryBuffer dataBuffer, HostMemoryBuffer validityBuffer,
                                  HostMemoryBuffer offsetBuffer, long child) {
    DeviceMemoryBuffer data = null;
    DeviceMemoryBuffer valid = null;
    DeviceMemoryBuffer offsets = null;
    if (dataBuffer != null) {
      long dataLen = rows * type.sizeInBytes;
      if (type == DType.STRING) {
        // This needs a different type
        dataLen = getEndStringOffset(rows - 1);
        if (dataLen == 0 && getNullCount() == 0) {
          // This is a work around to an issue where a column of all empty strings must have at
          // least one byte or it will not be interpreted correctly.
          dataLen = 1;
        }
      } else if (type == DType.LIST) {
        dataLen = dataBuffer.length;
      }
      data = DeviceMemoryBuffer.allocate(dataLen);
      data.copyFromHostBuffer(dataBuffer, 0, dataLen);
    }
    if (validityBuffer != null) {
      long validLen = ColumnVector.getNativeValidPointerSize((int) rows);
      valid = DeviceMemoryBuffer.allocate(validLen);
      valid.copyFromHostBuffer(validityBuffer, 0, validLen);
    }
    if (offsetBuffer != null) {
      long offsetsLen = OFFSET_SIZE * (rows + 1);
      offsets = DeviceMemoryBuffer.allocate(offsetsLen);
      offsets.copyFromHostBuffer(offsetBuffer, 0, offsetsLen);
    }
    ColumnVector ret = new ColumnVector(type, rows, nullCount, data, valid, offsets, child);
    return ret;
  }
  /////////////////////////////////////////////////////////////////////////////
  // DATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  public List getList(long rowIndex) throws Exception {
    return getListParent(rowIndex, 0);
  }
  public List getListParent(long rowIndex, int level) throws IOException {
    // check if list is further nested
    if (level < offHeap.offsets.size() - 2) {
      List retList = new ArrayList();
      HostMemoryBuffer mainOffset = offHeap.offsets.get(level);
      DType baseType = allTypes.get(allTypes.size() - 1);
      int start = mainOffset.getInt(rowIndex*DType.INT32.getSizeInBytes());
      int end = mainOffset.getInt((rowIndex+1)*DType.INT32.getSizeInBytes());
      for(int j = start;j < end;j++) {
        retList.add(getListParent(j, level + 1));
      }
      return retList;
    } else if (allTypes.get(allTypes.size() - 1) == DType.STRING) {
      HostMemoryBuffer mainOffset = offHeap.offsets.get(level);
      int start = mainOffset.getInt(rowIndex*DType.INT32.getSizeInBytes());
      int end = mainOffset.getInt((rowIndex+1)*DType.INT32.getSizeInBytes());
      HostMemoryBuffer strOffsets = offHeap.offsets.get(level + 1);
      List retList = new ArrayList();
      for (int index = start; index < end; index++) {
        int startStrOffset = strOffsets.getInt(index*DType.INT32.getSizeInBytes());
        int endStrOffset = strOffsets.getInt((index+1)*DType.INT32.getSizeInBytes());
        int size = endStrOffset-startStrOffset;
        byte[] rawData = new byte[endStrOffset-startStrOffset];
        if (size > 0) {
          offHeap.data.getBytes(rawData, 0, startStrOffset, size);
        }
        retList.add(new String(rawData));
      }
      return retList;
    } else { //Level 1 list
      HostMemoryBuffer mainOffset = offHeap.offsets.get(level);
      DType baseType = allTypes.get(allTypes.size() - 1);
      int start = mainOffset.getInt(rowIndex*DType.INT32.getSizeInBytes());
      int end = mainOffset.getInt((rowIndex+1)*DType.INT32.getSizeInBytes());
      int size = (end-start)*baseType.getSizeInBytes();
      byte[] rawData = new byte[size];
      if (size > 0) {
        offHeap.data.getBytes(rawData, 0, start*baseType.getSizeInBytes(), size);
      }

      for (int i =0; i < rawData.length;i++) {
        System.out.print((rawData[i]) + " ");
      }
      ByteArrayInputStream bais = new ByteArrayInputStream(rawData);
      DataInputStream dataInputStream = new DataInputStream(bais);
      List<Integer> list = new ArrayList<>();
      readToList(dataInputStream, list, baseType);
      return list;
    }
  }

  private void readToList(DataInputStream dataInputStream, List list, DType baseType) throws IOException {
    while (dataInputStream.available() > 0) {
      switch (baseType) {
        case INT32:
          list.add(dataInputStream.readInt());
          break;
        case INT64: list.add(dataInputStream.readLong());
          break;
        case FLOAT32: list.add(dataInputStream.readFloat());
          break;
        case FLOAT64: list.add(dataInputStream.readDouble());
          break;
        case INT8: list.add(dataInputStream.readByte());
          break;
        case INT16: list.add(dataInputStream.readShort());
          break;
        case BOOL8: list.add(dataInputStream.readBoolean());
          break;
        case STRING: list.add(dataInputStream.readUTF()); //cross check for utf8 etc.
          break;
        default: throw new UnsupportedOperationException("Do not support " + baseType);
      }
    }
  }

  /**
   * Check if the value at index is null or not.
   */
  public boolean isNull(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    if (hasValidityVector()) {
      if (nullCount.isPresent() && !hasNulls()) {
        return false;
      }
      return BitVectorHelper.isNull(offHeap.valid.get(0), index);
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
      return BitVectorHelper.isNull(offHeap.valid.get(0), index);
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
        srcBuffer = offHeap.valid.get(0);
        break;
      case OFFSET:
        srcBuffer = offHeap.offsets.get(0);
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
    return offHeap.offsets.get(0).getInt(index * 4);
  }

  /**
   * Get the ending byte offset for the string at index.
   */
  long getEndStringOffset(long index) {
    assert type == DType.STRING || type == DType.LIST;
    assert (index >= 0 && index < allRows.get(allRows.size() - 1)) : "index is out of range 0 <= " + index + " < " + rows;
    // The offsets has one more entry than there are rows.
    HostMemoryBuffer offset;
    if (type == DType.STRING) {
      offset = offHeap.offsets.get(0);
    } else {
      offset = offHeap.offsets.get(offHeap.offsets.size() - 1);
    }
    return offset.getInt((index + 1) * 4);
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
    int start = offHeap.offsets.get(0).getInt(index * OFFSET_SIZE);
    int size = offHeap.offsets.get(0).getInt((index + 1) * OFFSET_SIZE) - start;
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
    return new String(rawData, UTF_8);
  }

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  protected static final class OffHeapState extends MemoryCleaner.Cleaner {
    public HostMemoryBuffer data;
    public List<HostMemoryBuffer> valid = new ArrayList<>();
    public List<HostMemoryBuffer> offsets = new ArrayList<>();

    OffHeapState(HostMemoryBuffer data, List<HostMemoryBuffer> valid, List<HostMemoryBuffer> offsets) {
      this.data = data;
      this.valid = valid;
      this.offsets = offsets;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (data != null || valid != null || offsets != null) {
        try {
          ColumnVector.closeBuffers(data, valid.isEmpty() ? null : valid.get(0),
              offsets.isEmpty() ? null : offsets.get(0));
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
        for (HostMemoryBuffer validity : valid) {
          validity.noWarnLeakExpected();
        }
      }
      if (offsets != null) {
        for (HostMemoryBuffer offset : offsets) {
          offset.noWarnLeakExpected();
        }
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
        //TODO: fix for lists
        total += valid.get(0).length;
      }
      if (data != null) {
        total += data.length;
      }
      if (offsets != null) {
        total += offsets.get(0).length;
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

  //for lists
  public static Builder builder(DType type, DType baseType, int rows, int bufferLen) {
    return new Builder(type, baseType, rows, bufferLen);
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

  //special constructor for lists for simplicity
  public static HostColumnVector build(DType listType, DType baseType, int rows, long buffSize, Consumer<Builder> init) {
    try (HostColumnVector.Builder builder = builder(listType, baseType, rows, (int)buffSize)) {
      init.accept(builder);
      return builder.build();
    }
  }

  public static<T> HostColumnVector fromLists(DType baseType, List<T>... values) {
    int rows = values.length;
    long nullCount = 0;
    long bufferSize = 0;
    for (List s: values) {
      if (s == null) {
        nullCount++;
      } else {
        bufferSize += s.size()*baseType.getSizeInBytes();
      }
    }
    //TODO: ADD support for nulls
    return build(DType.LIST, baseType, rows, bufferSize, (b) -> {
      for (List s: values) {
        b.appendList(DType.LIST, baseType, 0, values.length + 1 , s);
      }
    });
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
        bufferSize += s.getBytes(UTF_8).length;
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
   * Build
   */
  public static final class Builder implements AutoCloseable {
    private final long rows;
    private final List<Long> allRows = new ArrayList<>();
    private final DType type;
    private final List<DType> allTypes = new ArrayList<>();
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private List<HostMemoryBuffer> allValids = new ArrayList<>();
    private List<HostMemoryBuffer> allOffsets = new ArrayList<>();
    private ArrayList<Integer> currentOffsets = new ArrayList<>();
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
      this.allTypes.add(type);
      this.rows = rows;
      this.allRows.add(rows);
      if (type == DType.STRING) {
        if (stringBufferSize <= 0) {
          // We need at least one byte or we will get NULL back for data
          stringBufferSize = 1;
        }
        this.data = HostMemoryBuffer.allocate(stringBufferSize);
        // The offsets are ints and there is 1 more than the number of rows.
        this.offsets = HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE);
        allOffsets.add(this.offsets);
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
      this.allTypes.add(type);
      this.rows = rows;
      this.allRows.add(rows);
      this.data = testData;
      this.valid = testValid;
      this.allValids.add(testValid);
    }

    //Right now a builder just for Lists - kept separate
    Builder(DType type, DType baseType, long rows, long stringBufferSize) {
      this.type = type;
      this.allTypes.add(type);
      this.rows = rows;
      this.allRows.add(rows);
      if (type == DType.LIST) {
        if (stringBufferSize <= 0) {
          // We need at least one byte or we will get NULL back for data
          stringBufferSize = 1;
        }
        this.data = HostMemoryBuffer.allocate(stringBufferSize);
      } else {
        throw new IllegalStateException("This Builder should not be used for non list types");
      }
    }

    //TODO: Needs clean up
    public Builder appendList(DType type, DType baseType, int level, int prevSize, List list) {
      if (list.isEmpty()) {
        throw new IllegalStateException("Cannot handle empty lists just yet");
      } else if (list.get(0) instanceof List) {
        if (allTypes.size() <= level + 1){
          allTypes.add(level + 1, DType.LIST);
        } else {
          allTypes.set(level + 1, DType.LIST);
        }
        int newSize = 0;
        List<List> tmpList = list;
        for(List insideList: tmpList) {
          newSize++;
        }
        if (allOffsets.size() <= level) {
          this.allOffsets.add(level, HostMemoryBuffer.allocate(prevSize * OFFSET_SIZE));
          this.allOffsets.get(level).setInt(0,0);
          this.currentOffsets.add(level, OFFSET_SIZE);
        }
        for(List insideList: tmpList) {
          appendList(type, baseType, level + 1, prevSize + newSize, insideList);
        }

        this.allOffsets.get(level).setInt(
            this.currentOffsets.get(level),
            this.allOffsets.get(level).getInt(
                this.currentOffsets.get(level)- OFFSET_SIZE) + list.size());
        this.currentOffsets.set(level, this.currentOffsets.get(level)+OFFSET_SIZE);
        if (level +1 >= allRows.size()) {
          allRows.add(level + 1, (long)tmpList.size());
        } else {
          allRows.set(level + 1, allRows.get(level + 1) + (long) tmpList.size());
        }
        return this;
      } else {
        if (allTypes.size() <= level + 1){
          allTypes.add(level + 1, baseType);
        } else {
          allTypes.set(level + 1, baseType);
        }
        int baseTypeSizeInBytes = baseType.getSizeInBytes();
        assert list != null : "appendNull must be used to append null strings";
        int length = list.size();
        // just for lists we want to throw a real exception if we would overrun the buffer
        long oldLen = data.getLength();
        long newLen = oldLen;
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        byte[] listBytes = null;
        try {
          writeLists(dos, list, baseType);
          dos.flush();
          listBytes = bos.toByteArray();
        } catch (IOException e) {
          // ignore close exception
        }
        if (baseType != DType.STRING) {
          while (currentListCount * baseTypeSizeInBytes + list.size() * baseTypeSizeInBytes > newLen) {
            newLen *= 2;
          }
        } else {
          while (newLen - currentStringByteIndex < listBytes.length) {
            newLen = newLen + listBytes.length;
          }
        }
        if (newLen > Integer.MAX_VALUE) {
          throw new IllegalStateException("A string buffer is not supported over 2GB in size");
        }
        if (newLen != oldLen) {
          // need to grow the size of the buffer.
          HostMemoryBuffer newData = HostMemoryBuffer.allocate(newLen);
          try {
            if (baseType != DType.STRING) {
              newData.copyFromHostBuffer(0, data, 0, currentListCount * baseTypeSizeInBytes);
            } else {
              newData.copyFromHostBuffer(0, data, 0, currentStringByteIndex);
            }
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
          for (int i = 0; i < listBytes.length; i++) {
            System.out.print((listBytes[i]) + " ");
          }
          if (baseType != DType.STRING) {
            data.setBytes(currentListCount * baseTypeSizeInBytes, listBytes, 0, listBytes.length);
          } else {
            data.setBytes(currentStringByteIndex, listBytes, 0, listBytes.length);
          }
          byte[] tmpArr = new byte[(int) data.length];
          data.getBytes(tmpArr, 0, 0, tmpArr.length);
          for (int i = 0; i < tmpArr.length; i++) {
            System.out.print((tmpArr[i]) + " ");
          }
        }
        currentListCount += length;
        currentListIndex++;
        if (allOffsets.size() <= level) {
          this.allOffsets.add(level, HostMemoryBuffer.allocate(prevSize * OFFSET_SIZE));
          allOffsets.get(level).setInt(0, 0);
          this.currentOffsets.add(level, OFFSET_SIZE);
        }
        this.allOffsets.get(level).setInt(this.currentOffsets.get(level), currentListCount);
        this.currentOffsets.set(level,this.currentOffsets.get(level)+OFFSET_SIZE);
        while (this.allRows.size() <= level + 1) {
          this.allRows.add(0l);
        }
        this.allRows.set(level + 1, (long) currentListCount);
      }
      if (baseType == DType.STRING) {
        if (!needToAdd) {
          //TODO: Fix this hard code
          allOffsets.add(HostMemoryBuffer.allocate(128));
          allOffsets.get(allOffsets.size() - 1).setInt(0, 0);
          needToAdd = true;
        }
        //////////////
        for (Object item : list) {
          String strItem = (String)item;
          int strLen = strItem.getBytes(UTF_8).length;
          for (int k =0;k < strItem.getBytes(UTF_8).length;k++) {
            System.out.print(strItem.getBytes(UTF_8)[k]);
          }
          currentStringByteIndex += strLen;
          currentIndex++;
          allOffsets.get(allOffsets.size() - 1).setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
        }
        //TODO: Fix this hard code
        printOffsetBuffer(allOffsets.size() - 1, 10);

      }
      return this;
    }

    private void printOffsetBuffer(int level, int prevSize) {
      byte[] offsetBytes = new byte[prevSize * OFFSET_SIZE];
      allOffsets.get(level).getBytes(offsetBytes, 0, 0, offsetBytes.length);
      for (int i = 0; i < offsetBytes.length; i++) {
        System.out.print(offsetBytes[i]);
      }
    }


    private void writeLists(DataOutputStream dos, List list, DType baseType) throws IOException {
      for (int i = 0; i < list.size(); i++) {
        switch (baseType) {
          case INT32:
            dos.writeInt((int) list.get(i));
            break;
          case INT64: dos.writeLong((long) list.get(i));
            break;
          case FLOAT32: dos.writeFloat((float) list.get(i));
            break;
          case FLOAT64: dos.writeDouble((double) list.get(i));
            break;
          case INT8: dos.writeByte((byte) list.get(i));
            break;
          case INT16: dos.writeShort((short) list.get(i));
            break;
          case BOOL8: dos.writeBoolean((boolean) list.get(i));
            break;
          case STRING: dos.writeBytes((String) list.get(i)); //cross check for utf8 etc.
            break;
          default: throw new UnsupportedOperationException("Do not support " + baseType);
        }
      }
      dos.flush();
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
      return appendUTF8String(value.getBytes(UTF_8));
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
        BitVectorHelper.append(columnVector.offHeap.valid.get(0), valid, currentIndex,
            columnVector.rows);
        nullCount += otherNc;
      }
      currentIndex += columnVector.rows;
      return this;
    }

    private void allocateBitmaskAndSetDefaultValues() {
      long bitmaskSize = ColumnVector.getNativeValidPointerSize((int) rows);
      valid = (HostMemoryBuffer.allocate(bitmaskSize));
      allValids.add(valid);
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
      if (built) {
        throw new IllegalStateException("Cannot reuse a builder.");
      }
      if (type != DType.LIST) {
        HostColumnVector cv = new HostColumnVector(type,
            currentIndex, Optional.of(nullCount), data, valid, offsets);
        built = true;
        return cv;
      } else {
        HostColumnVector lcv = new HostColumnVector(allTypes, allRows, Optional.of(nullCount),
            data, allValids, allOffsets);
        built = true;
        return lcv;
      }
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
