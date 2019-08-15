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
import java.util.function.Consumer;
import java.util.stream.IntStream;

/**
 * A Column Vector. This class represents the immutable vector of data.  This class holds
 * references to off heap memory and is reference counted to know when to release it.  Call
 * close to decrement the reference count when you are done with the column, and call inRefCount
 * to increment the reference count.
 */
public final class ColumnVector implements AutoCloseable, BinaryOperable {
  /**
   * The size in bytes of an offset entry
   */
  static final int OFFSET_SIZE = DType.INT32.sizeInBytes;
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
   * Wrap an existing on device gdf_column with the corresponding ColumnVector.
   */
  ColumnVector(long nativePointer) {
    assert nativePointer != 0;
    MemoryCleaner.register(this, offHeap);
    offHeap.nativeCudfColumnHandle = nativePointer;
    this.type = getDType(nativePointer);
    offHeap.setHostData(null);
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
    this.offHeap.setDeviceData(new BufferEncapsulator<>(data, valid, null));
    this.refCount = 0;
    incRefCountInternal(true);
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
   *                           entry. It should be (rows + 1) ints.
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
    MemoryCleaner.register(this, offHeap);
    offHeap.setHostData(new BufferEncapsulator(hostDataBuffer, hostValidityBuffer, offsetBuffer));
    offHeap.setDeviceData(null);
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;
    refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * Create a new column vector based off of data already on the device.
   * @param type the type of the vector
   * @param tsTimeUnit the unit of time for this vector
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
   * @param resetOffsetsFromFirst if true and type is a string or a string_category then when
   *                              unpacking the offsets, the initial offset will be reset to
   *                              0 and all other offsets will be updated to be relative to that
   *                              new 0.  This is used after serializing a partition, when the
   *                              offsets were not updated prior to the serialization.
   */
  ColumnVector(DType type, TimeUnit tsTimeUnit, long rows,
               long nullCount, DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer,
               HostMemoryBuffer offsetBuffer, boolean resetOffsetsFromFirst) {
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

    MemoryCleaner.register(this, offHeap);
    offHeap.setHostData(null);
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;

    if (type == DType.STRING || type == DType.STRING_CATEGORY) {
      if (type == DType.STRING_CATEGORY) {
        offHeap.setDeviceData(new BufferEncapsulator(DeviceMemoryBuffer.allocate(rows * type.sizeInBytes), validityBuffer, null));
      } else {
        offHeap.setDeviceData(new BufferEncapsulator(null, validityBuffer, null));
      }
      // In the case of STRING and STRING_CATEGORY the gdf_column holds references
      // to the device data that the java code does not, so we will not be lazy about
      // creating the gdf_column instance.
      offHeap.nativeCudfColumnHandle = allocateCudfColumn();

      cudfColumnViewStrings(offHeap.nativeCudfColumnHandle,
          dataBuffer.getAddress(),
          false,
          offsetBuffer.getAddress(),
          resetOffsetsFromFirst,
          nullCount > 0 ? offHeap.getDeviceData().valid.getAddress() : 0,
          offHeap.getDeviceData().data == null ? 0 : offHeap.getDeviceData().data.getAddress(),
          (int) rows, type.nativeId,
          (int) getNullCount());
      dataBuffer.close();
      offsetBuffer.close();
    } else {
      offHeap.setDeviceData(new BufferEncapsulator(dataBuffer, validityBuffer, null));
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
  public final void noWarnLeakExpected() {
    offHeap.noWarnLeakExpected();
    if (offHeap.getHostData() != null) {
      offHeap.getHostData().noWarnLeakExpected();
    }
    if (offHeap.getDeviceData() != null) {
      offHeap.getDeviceData().noWarnLeakExpected();
    }
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
        ", hostData=" + offHeap.getHostData() +
        ", deviceData=" + offHeap.getDeviceData() +
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
    return incRefCountInternal(false);
  }

  private ColumnVector incRefCountInternal(boolean isFirstTime) {
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
   * Retrieve the number of characters in each string. Null strings will have value of null.
   *
   * @return ColumnVector holding length of string at index 'i' in the original vector
   */
  public ColumnVector getLengths() {
    assert DType.STRING == type : "length only available for String type";
    return new ColumnVector(cudfLengths(getNativeCudfColumnAddress()));
  }

  /**
   * Compute the 32 bit hash of a vector.
   *
   * @return the 32 bit hash.
   */
  public ColumnVector hash() {
    return new ColumnVector(hash(getNativeCudfColumnAddress(), HashFunction.MURMUR3.nativeId));
  }

  /**
   * Compute a specific hash of a vector. String are not supported, if you need a hash of a string,
   * you can use the generic hash, which does not guarantee what kind of hash is used.
   * @param func the has function to use.
   * @return the 32 bit hash.
   */
  public ColumnVector hash(HashFunction func) {
    assert type != DType.STRING && type != DType.STRING_CATEGORY : "Strings are not supported for specific hash functions";
    return new ColumnVector(hash(getNativeCudfColumnAddress(), func.nativeId));
  }

  /**
   * Compute the MURMUR3 hash of the column. Strings are not supported.
   */
  public ColumnVector murmur3() {
    return hash(HashFunction.MURMUR3);
  }

  /**
   * Compute the IDENTITY hash of the column. Strings are not supported.
   */
  public ColumnVector identityHash() {
    return hash(HashFunction.IDENTITY);
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
   * Returns this column's current refcount
   */
  int getRefCount() {
    return refCount;
  }

  /**
   * Retrieve the number of bytes for each string. Null strings will have value of null.
   *
   * @return ColumnVector, where each element at i = byte count of string at index 'i' in the original vector
   */
  public ColumnVector getByteCount() {
    assert type == DType.STRING : "type has to be a String";
    return new ColumnVector(cudfByteCount(getNativeCudfColumnAddress()));
  }

  /**
   * Returns if the vector has a validity vector allocated or not.
   */
  public boolean hasValidityVector() {
    boolean ret;
    if (offHeap.getHostData() != null) {
      ret = (offHeap.getHostData().valid != null);
    } else {
      ret = (offHeap.getDeviceData().valid != null);
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
    if (offHeap.getDeviceData() == null && rows != 0) {
      if (refCount <= 0) {
        throw new IllegalStateException("Vector was already closed.");
      }
      throw new IllegalStateException("Vector not on Device");
    }
  }

  private void checkHasHostData() {
    if (offHeap.getHostData() == null && rows != 0) {
      throw new IllegalStateException("Vector not on Host");
    }
  }

  /**
   * Be sure the data is on the device.
   */
  public final void ensureOnDevice() {
    if (offHeap.getDeviceData() == null && rows != 0) {
      checkHasHostData();

      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
        assert (offHeap.getHostData().offsets != null);
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
        deviceDataBuffer = DeviceMemoryBuffer.allocate(offHeap.getHostData().data.getLength());
      }

      boolean needsCleanup = true;
      try {
        if (hasNulls()) {
          deviceValidityBuffer = DeviceMemoryBuffer.allocate(offHeap.getHostData().valid.getLength());
        }
        offHeap.setDeviceData(new BufferEncapsulator(deviceDataBuffer, deviceValidityBuffer, null));
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

      if (offHeap.getDeviceData().valid != null) {
        offHeap.getDeviceData().valid.copyFromHostBuffer(offHeap.getHostData().valid);
      }

      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
        // In the case of STRING and STRING_CATEGORY the gdf_column holds references
        // to the device data that the java code does not, so we will not be lazy about
        // creating the gdf_column instance.
        offHeap.nativeCudfColumnHandle = allocateCudfColumn();
        cudfColumnViewStrings(offHeap.nativeCudfColumnHandle,
            offHeap.getHostData().data.getAddress(),
            true, offHeap.getHostData().offsets.getAddress(),
            false, offHeap.getHostData().valid == null ? 0 : offHeap.getDeviceData().valid.getAddress(),
            offHeap.getDeviceData().data == null ? 0 : offHeap.getDeviceData().data.getAddress(),
            (int) rows, type.nativeId,
            (int) getNullCount());
      } else {
        offHeap.getDeviceData().data.copyFromHostBuffer(offHeap.getHostData().data);
      }
    }
  }

  /**
   * Be sure the data is on the host.
   */
  public final void ensureOnHost() {
    if (offHeap.getHostData() == null && rows != 0) {
      checkHasDeviceData();

      HostMemoryBuffer hostDataBuffer = null;
      HostMemoryBuffer hostValidityBuffer = null;
      HostMemoryBuffer hostOffsetsBuffer = null;
      boolean needsCleanup = true;
      try {
        if (offHeap.getDeviceData().valid != null) {
          hostValidityBuffer = HostMemoryBuffer.allocate(offHeap.getDeviceData().valid.getLength());
        }
        if (type == DType.STRING || type == DType.STRING_CATEGORY) {
          long[] vals = getStringDataAndOffsetsBack(getNativeCudfColumnAddress());
          hostDataBuffer = new HostMemoryBuffer(vals[0], vals[1]);
          hostOffsetsBuffer = new HostMemoryBuffer(vals[2], vals[3]);
        } else {
          hostDataBuffer = HostMemoryBuffer.allocate(offHeap.getDeviceData().data.getLength());
        }

        offHeap.setHostData(new BufferEncapsulator(hostDataBuffer, hostValidityBuffer,
            hostOffsetsBuffer));
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
        offHeap.getHostData().data.copyFromDeviceBuffer(offHeap.getDeviceData().data);
      }
      if (offHeap.getHostData().valid != null) {
        offHeap.getHostData().valid.copyFromDeviceBuffer(offHeap.getDeviceData().valid);
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
      return BitVectorHelper.isNull(offHeap.getHostData().valid, index);
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
      return BitVectorHelper.isNull(offHeap.getHostData().valid, index);
    }
    return false;
  }

  enum BufferType {
    VALIDITY,
    OFFSET,
    DATA
  }

  void copyHostBufferBytes(byte[] dst, int dstOffset, BufferType src, long srcOffset,
                           int length) {
    assert dstOffset >= 0;
    assert srcOffset >= 0;
    assert length >= 0;
    assert dstOffset + length <= dst.length;
    HostMemoryBuffer srcBuffer;
    switch(src) {
      case VALIDITY:
        srcBuffer = offHeap.getHostData().valid;
        break;
      case OFFSET:
        srcBuffer = offHeap.getHostData().offsets;
        break;
      case DATA:
        srcBuffer = offHeap.getHostData().data;
        break;
      default:
        throw new IllegalArgumentException(src + " is not a supported buffer type.");
    }

    assert srcOffset + length <= srcBuffer.length;
    UnsafeMemoryAccessor.getBytes(dst, dstOffset,
        srcBuffer.getAddress() + srcOffset, length);
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is not null, and FALSE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public ColumnVector isNotNull() {
    return validityAsBooleanVector();
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * FALSE for any entry that is not null, and TRUE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public ColumnVector isNull() {
    ColumnVector res = null;
    try (ColumnVector boolValidity = validityAsBooleanVector()) {
      res = boolValidity.not();
    }
    return res;
  }

  /**
   * Returns a ColumnVector with any null values replaced with a scalar.
   *
   * @param scalar - Scalar value to use as replacement
   * @return - ColumnVector with nulls replaced by scalar
   */
  public ColumnVector replaceNulls(Scalar scalar) {
    return new ColumnVector(Cudf.replaceNulls(this, scalar));
  }

  /*
   * Returns the validity mask as a boolean vector with FALSE for nulls, 
   * and TRUE otherwise.
   */
  private ColumnVector validityAsBooleanVector() {
    if (getRowCount() == 0) {
      return ColumnVector.fromBoxedBooleans();
    }

    ColumnVector cv = ColumnVector.fromScalar(Scalar.fromBool(true), (int)getRowCount());

    if (getNullCount() == 0) {
      // we are done, all not null
      return cv;
    }

    ColumnVector result = null;

    try {
      // we are using the device validity bitmask, lets check
      // we are in the device 
      checkHasDeviceData();

      // Apply the validity mask to cv's native column vector
      // Warning: This is sharing the validity vector from the current column
      // with the new column created with fromScalar temporarily.
      cudfColumnViewAugmented(
          cv.offHeap.nativeCudfColumnHandle,
          cv.offHeap.getDeviceData().data.address,
          this.offHeap.getDeviceData().valid.address,
          (int) this.getRowCount(),
          DType.BOOL8.nativeId,
          (int) getNullCount(),
          TimeUnit.NONE.getNativeId());

      // just to keep java in-sync
      cv.nullCount = getNullCount();

      // replace nulls with FALSE
      result = cv.replaceNulls(Scalar.fromBool(false));
    } finally {
      // cleanup
      if (cv != null) {
        if (cv.offHeap.getDeviceData().data != null) {
          cv.offHeap.getDeviceData().data.close(); // don't need this anymore
        }
        cv.offHeap.setDeviceData(null); // .valid is managed by the current column vector, so
                                      // don't let it get closed
        cv.close(); // clean up other resources
      }
    }
    return result;
  }

  /**
   * Generic type independent asserts when getting a value from a single index.
   * @param index where to get the data from.
   */
  private void assertsForGet(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert offHeap.getHostData() != null : "data is not on the host";
    assert !isNull(index) : " value at " + index + " is null";
  }

  /**
   * Get the value at index.
   */
  public byte getByte(long index) {
    assert type == DType.INT8 || type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.getHostData().data.getByte(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final short getShort(long index) {
    assert type == DType.INT16;
    assertsForGet(index);
    return offHeap.getHostData().data.getShort(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final int getInt(long index) {
    assert type == DType.INT32 || type == DType.DATE32;
    assertsForGet(index);
    return offHeap.getHostData().data.getInt(index * type.sizeInBytes);
  }

  /**
   * Get the starting byte offset for the string at index
   */
  long getStartStringOffset(long index) {
    assert type == DType.STRING_CATEGORY || type == DType.STRING;
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert offHeap.getHostData() != null : "data is not on the host";
    return offHeap.getHostData().offsets.getInt(index * 4);
  }

  /**
   * Get the ending byte offset for the string at index.
   */
  long getEndStringOffset(long index) {
    assert type == DType.STRING_CATEGORY || type == DType.STRING;
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert offHeap.getHostData() != null : "data is not on the host";
    // The offsets has one more entry than there are rows.
    return offHeap.getHostData().offsets.getInt((index + 1) * 4);
  }

  /**
   * Get the value at index.
   */
  public final long getLong(long index) {
    assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
    assertsForGet(index);
    return offHeap.getHostData().data.getLong(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final float getFloat(long index) {
    assert type == DType.FLOAT32;
    assertsForGet(index);
    return offHeap.getHostData().data.getFloat(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final double getDouble(long index) {
    assert type == DType.FLOAT64;
    assertsForGet(index);
    return offHeap.getHostData().data.getDouble(index * type.sizeInBytes);
  }

  /**
   * Get the boolean value at index
   */
  public final boolean getBoolean(long index) {
    assert type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.getHostData().data.getBoolean(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.  This API is slow as it has to translate the
   * string representation.  Please use it with caution.
   */
  public String getJavaString(long index) {
    assert type == DType.STRING || type == DType.STRING_CATEGORY;
    assertsForGet(index);
    int start = offHeap.getHostData().offsets.getInt(index * OFFSET_SIZE);
    int size = offHeap.getHostData().offsets.getInt((index + 1) * OFFSET_SIZE) - start;
    byte[] rawData = new byte[size];
    if (size > 0) {
      offHeap.getHostData().data.getBytes(rawData, 0, start, size);
    }
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
    return slice(ColumnVector.fromInts(indices));
  }

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
  public ColumnVector[] slice(ColumnVector indices) {
    long[] nativeHandles = cudfSlice(this.getNativeCudfColumnAddress(), indices.getNativeCudfColumnAddress());
    ColumnVector[] columnVectors = new ColumnVector[nativeHandles.length];
    IntStream.range(0, nativeHandles.length).forEach(i -> columnVectors[i] = new ColumnVector(nativeHandles[i]));
    return columnVectors;
  }

  /**
   * Fill the current vector (note this is in-place) with a Scalar value.
   *
   * String categories are not supported by cudf::fill. Additionally, Scalar
   * does not support Strings either.
   *
   * @param scalar - Scalar value to replace row with
   * @throws IllegalArgumentException
   */
  // package private for testing
  void fill(Scalar scalar) throws IllegalArgumentException {
    assert scalar.getType() == this.getType();

    if (this.getType() == DType.STRING || this.getType() == DType.STRING_CATEGORY){
      throw new IllegalStateException("DType of STRING, or STRING_CATEGORY not supported");
    }

    if (this.getRowCount() == 0) {
      return; // no rows to fill
    }

    checkHasDeviceData();

    BufferEncapsulator<DeviceMemoryBuffer> newDeviceData = null;
    boolean needsCleanup = true;

    try {
      if (!scalar.isValid()) {
        if (getNullCount() == getRowCount()) {
          //current vector has all nulls, and we are trying to set it to null.
          return;
        }
        this.nullCount = rows;

        if (offHeap.getDeviceData().valid == null) {
          long validitySizeInBytes = BitVectorHelper.getValidityAllocationSizeInBytes(rows);
          // scalar is null and vector doesn't have a validity mask. Create a validity mask.
          newDeviceData = new BufferEncapsulator<DeviceMemoryBuffer>(
              this.offHeap.getDeviceData().data,
              DeviceMemoryBuffer.allocate(validitySizeInBytes),
              null);
          this.offHeap.setDeviceData(newDeviceData);
        } else {
          newDeviceData = this.offHeap.getDeviceData();
        }

        // the buffer encapsulator is the owner of newDeviceData, no need to clear
        needsCleanup = false;

        Cuda.memset(newDeviceData.valid.getAddress(), (byte) 0x00,
            BitVectorHelper.getValidityLengthInBytes(rows));

        // set the validity pointer
        cudfColumnViewAugmented(
            this.getNativeCudfColumnAddress(),
            newDeviceData.data.address,
            newDeviceData.valid.address,
            (int) this.getRowCount(),
            this.getType().nativeId,
            (int) this.nullCount,
            this.getTimeUnit().getNativeId());

      } else {
        this.nullCount = 0;
        newDeviceData = this.offHeap.getDeviceData();
        needsCleanup = false; // the data came from upstream

        // if we are now setting the vector to a non-null, we need to
        // close out the validity vector
        if (newDeviceData.valid != null){
          newDeviceData.valid.close();
          newDeviceData = new BufferEncapsulator<DeviceMemoryBuffer>(
              newDeviceData.data, null, null);
          this.offHeap.setDeviceData(newDeviceData);
        }

        // set the validity pointer
        cudfColumnViewAugmented(
            this.getNativeCudfColumnAddress(),
            newDeviceData.data.address,
            0,
            (int) this.getRowCount(),
            this.getType().nativeId,
            (int) nullCount,
            this.getTimeUnit().getNativeId());

        Cudf.fill(this, scalar);
      }

      // at this stage, host offHeap is no longer meaningful
      // if we had hostData, reset it with a fresh copy from device
      if (this.offHeap.getHostData() != null) {
        this.offHeap.getHostData().close();
        this.offHeap.setHostData(null);
        this.ensureOnHost();
      }
    } finally {
      if (needsCleanup) {
        if (newDeviceData != null) {
          // we allocated a bit vector, but we errored out
          newDeviceData.valid.close();
        }
      }
    }
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
    return reduce(ReductionOp.SUM, outType);
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
    return reduce(ReductionOp.MIN, outType);
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
    return reduce(ReductionOp.MAX, outType);
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
    return reduce(ReductionOp.PRODUCT, outType);
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
    return reduce(ReductionOp.SUMOFSQUARES, outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a FLOAT64 scalar unless the column type is FLOAT32 then
   * a FLOAT32 scalaris returned. Null's are not counted as an element
   * of the column when calculating the standard deviation.
   */
  public Scalar standardDeviation() {
    if(type != DType.FLOAT32)
      standardDeviation(DType.FLOAT64);
    return standardDeviation(type);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a scalar of the specified type. Null's are not counted as
   * an element of the column when calculating the standard deviation.
   */
  public Scalar standardDeviation(DType outType) {
    return reduce(ReductionOp.STD, outType);
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
  public Scalar reduce(ReductionOp op) {
    return reduce(op, type);
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
  public Scalar reduce(ReductionOp op, DType outType) {
    return Cudf.reduce(this, op, outType);
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
        dataAddr = offHeap.getDeviceData().data.getAddress();
        if (offHeap.getDeviceData().valid != null) {
          validAddr = offHeap.getDeviceData().valid.getAddress();
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

  private native static long cudfByteCount(long cudfColumnHandle) throws CudfException;

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

  private native long[] cudfSlice(long nativeHandle, long indices) throws CudfException;

  /**
   * Translate the host side string representation of strings into the device side representation
   * and populate the cudfColumn with it.
   * @param cudfColumnHandle native handle of gdf_column.
   * @param dataPtr          Pointer to string data either on the host or the device.
   * @param dataPtrOnHost    true if dataPtr is on the host. false if it is on the device.
   * @param hostOffsetsPtr   Pointer to offsets data on the host.
   * @param resetOffsetsToZero true if the offsets should be reset to start at 0.
   * @param deviceValidPtr   Pointer to validity bitmask on the device.
   * @param deviceOutputDataPtr Pointer to where the int category data will be stored for
   *                            STRING_CATEGORY. Should be 0 for STRING.
   * @param numRows          Number of rows in the column.
   * @param dtype            Data type of the column. In this case must be STRING or
   *                         STRING_CATEGORY
   * @param nullCount        The number of non-valid elements in the validity bitmask.
   */
  private static native void cudfColumnViewStrings(long cudfColumnHandle, long dataPtr,
                                                   boolean dataPtrOnHost, long hostOffsetsPtr,
                                                   boolean resetOffsetsToZero, long deviceValidPtr,
                                                   long deviceOutputDataPtr,
                                                   int numRows, int dtype, int nullCount);

  private native Scalar exactQuantile(long cudfColumnHandle, int quantileMethod, double quantile) throws CudfException;

  private native Scalar approxQuantile(long cudfColumnHandle, double quantile) throws CudfException;

  private static native long cudfLengths(long cudfColumnHandle) throws CudfException;

  private static native long hash(long cudfColumnHandle, int nativeHashId) throws CudfException;

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

    /**
     * This is a really ugly API, but it is possible that the lifecycle of a column of
     * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
     * tracking code that this is expected for this column, and big scary warnings should
     * not be printed when this happens.
     */
    public void noWarnLeakExpected() {
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
  }

  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  protected static final class OffHeapState extends MemoryCleaner.Cleaner {
    private BufferEncapsulator<HostMemoryBuffer> hostData;
    private BufferEncapsulator<DeviceMemoryBuffer> deviceData;
    private long nativeCudfColumnHandle = 0;

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (getHostData() != null) {
        getHostData().close();
        setHostData(null);
        neededCleanup = true;
      }
      if (getDeviceData() != null) {
        getDeviceData().close();
        setDeviceData(null);
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

    public BufferEncapsulator<HostMemoryBuffer> getHostData() {
      return hostData;
    }

    public void setHostData(BufferEncapsulator<HostMemoryBuffer> hostData) {
      if (isLeakExpected() && hostData != null) {
        hostData.noWarnLeakExpected();
      }
      this.hostData = hostData;
    }

    public BufferEncapsulator<DeviceMemoryBuffer> getDeviceData() {
      return deviceData;
    }

    public void setDeviceData(BufferEncapsulator<DeviceMemoryBuffer> deviceData) {
      if (isLeakExpected() && deviceData != null) {
        deviceData.noWarnLeakExpected();
      }
      this.deviceData = deviceData;
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
    return new Builder(type, TimeUnit.NONE, rows, 0);
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
    return new Builder(type, tsTimeUnit, rows, 0);
  }

  /**
   * Create a new Builder to hold the specified number of rows and with enough space to hold the
   * given amount of string data. Be sure to close the builder when done with it. Please try to
   * use {@see #build(int, int, Consumer)} instead to avoid needing to close the builder.
   * @param type the type of vector to build.
   * @param rows the number of rows this builder can hold
   * @param stringBufferSize the size of the string buffer to allocate.
   * @return the builder to use.
   */
  public static Builder builder(DType type, int rows, long stringBufferSize) {
    assert type == DType.STRING_CATEGORY || type == DType.STRING;
    return new Builder(type, TimeUnit.NONE, rows, stringBufferSize);
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

  public static ColumnVector build(DType type, int rows, long stringBufferSize, Consumer<Builder> init) {
    try (Builder builder = builder(type, rows, stringBufferSize)) {
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
    assert type == DType.STRING || type == DType.STRING_CATEGORY;
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
      return build(type, rows, bufferSize, (b) -> b.appendBoxed(values));
    }
    return build(type, rows, bufferSize, (b) -> {
      for (String s: values) {
        b.append(s);
      }
    });
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
   * Create a new vector of length rows, where each row is filled with the Scalar's
   * value
   * @param scalar - Scalar to use to fill rows
   * @param rows - Number of rows in the new ColumnVector
   * @return - new ColumnVector
   */
  public static ColumnVector fromScalar(Scalar scalar, int rows) {
    if (scalar.getType() == DType.STRING || scalar.getType() == DType.STRING_CATEGORY) {
      throw new IllegalArgumentException("STRING and STRING_CATEGORY are not supported scalars");
    }
    DeviceMemoryBuffer dataBuffer = null;
    ColumnVector cv = null;
    boolean needsCleanup = true;

    try {
      dataBuffer = DeviceMemoryBuffer.allocate(scalar.type.sizeInBytes * rows);

      cv = new ColumnVector(
          scalar.getType(),
          scalar.getTimeUnit(),
          rows,
          0,
          dataBuffer,
          null,
          null, false);

      // null this out as cv is the owner, and will be closed
      // when cv closes in case of failure
      dataBuffer = null;

      cudfColumnViewAugmented(
          cv.getNativeCudfColumnAddress(),
          cv.offHeap.getDeviceData().data.address,
          0,
          (int) cv.getRowCount(),
          cv.getType().nativeId,
          0,
          cv.getTimeUnit().getNativeId());

      cv.fill(scalar);

      needsCleanup = false;
      return cv;
    } finally {
      if (needsCleanup) {
        if (dataBuffer != null) {
          dataBuffer.close();
        }
        if (cv != null) {
          cv.close();
        }
      }
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
    long[] columnHandles = new long[columns.length];
    for (int i = 0; i < columns.length; ++i) {
      columnHandles[i] = columns[i].getNativeCudfColumnAddress();
    }
    return new ColumnVector(concatenate(columnHandles));
  }

  /**
   * Calculate the quantile of this ColumnVector
   * @param method   the method used to calculate the quantile
   * @param quantile the quantile value [0,1]
   * @return the quantile as double. The type can be changed in future
   */
  public Scalar exactQuantile(QuantileMethod method, double quantile) {
    return exactQuantile(this.getNativeCudfColumnAddress(), method.nativeId, quantile);
  }

  /**
   * Calculate the approximate quantile of this ColumnVector
   * @param quantile the quantile value [0,1]
   * @return the quantile, with the same type as this object
   */
  public Scalar approxQuantile(double quantile) {
    return approxQuantile(this.getNativeCudfColumnAddress(), quantile);
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
    private HostMemoryBuffer offsets;
    private long currentIndex = 0;
    private long nullCount;
    private long stringBufferSize = 0;
    private int currentStringByteIndex = 0;
    private boolean built;

    /**
     * Create a builder with a buffer of size rows
     * @param type       datatype
     * @param tsTimeUnit for TIMESTAMP the unit of time it is storing.
     * @param rows       number of rows to allocate.
     * @param stringBufferSize the size of the string data buffer if we are
     *                         working with Strings.  It is ignored otherwise.
     */
    Builder(DType type, TimeUnit tsTimeUnit, long rows, long stringBufferSize) {
      this.type = type;
      this.tsTimeUnit = tsTimeUnit;
      this.rows = rows;
      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
        if (stringBufferSize <= 0) {
          // We need at least one byte or we will get NULL back for data
          stringBufferSize = 1;
        }
        this.data = HostMemoryBuffer.allocate(stringBufferSize);
        // The offsets are ints and there is 1 more than the number of rows.
        this.offsets = HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE);
        // The first offset is always 0
        this.offsets.setInt(0, 0);
        this.stringBufferSize = stringBufferSize;
      } else {
        this.data = HostMemoryBuffer.allocate(rows * type.sizeInBytes);
      }
    }

    /**
     * Create a builder with a buffer of size rows (for testing ONLY).
     * @param type       datatype
     * @param tsTimeUnit for TIMESTAMP the unit of time it is storing.
     * @param rows       number of rows to allocate.
     * @param testData   a buffer to hold the data (should be large enough to hold rows entries).
     * @param testValid  a buffer to hold the validity vector (should be large enough to hold
     *                   rows entries or is null).
     * @param testOffsets a buffer to hold the offsets for strings and string categories.
     */
    Builder(DType type, TimeUnit tsTimeUnit, long rows, HostMemoryBuffer testData,
            HostMemoryBuffer testValid, HostMemoryBuffer testOffsets) {
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
      assert type == DType.STRING_CATEGORY || type == DType.STRING;
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
      assert type == DType.INT8 || type == DType.BOOL8;
      data.setBytes(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(short... values) {
      assert type == DType.INT16;
      assert (values.length + currentIndex) <= rows;
      data.setShorts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(int... values) {
      assert (type == DType.INT32 || type == DType.DATE32);
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(long... values) {
      assert type == DType.INT64 || type == DType.DATE64 || type == DType.TIMESTAMP;
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

    /**
     * Append this vector to the end of this vector
     * @param columnVector - Vector to be added
     * @return - The ColumnVector based on this builder values
     */
    public final Builder append(ColumnVector columnVector) {
      assert columnVector.rows <= (rows - currentIndex);
      assert columnVector.type == type;
      assert columnVector.offHeap.getHostData() != null;

      if (type == DType.STRING_CATEGORY || type == DType.STRING) {
        throw new UnsupportedOperationException(
            "Appending a string column vector client side is not currently supported");
      } else {
        data.copyFromHostBuffer(currentIndex * type.sizeInBytes, columnVector.offHeap.getHostData().data,
            0L,
            columnVector.getRowCount() * type.sizeInBytes);
      }

      if (columnVector.nullCount != 0) {
        if (valid == null) {
          allocateBitmaskAndSetDefaultValues();
        }
        //copy values from intColumnVector to this
        BitVectorHelper.append(columnVector.offHeap.getHostData().valid, valid, currentIndex,
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
      if (type == DType.STRING || type == DType.STRING_CATEGORY) {
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
     * Finish and create the immutable ColumnVector.
     */
    public final ColumnVector build() {
      if (built) {
        throw new IllegalStateException("Cannot reuse a builder.");
      }
      ColumnVector cv = new ColumnVector(type, tsTimeUnit,
          currentIndex, nullCount, data, valid, offsets);
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
          currentIndex, nullCount, data, valid, offsets);
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
