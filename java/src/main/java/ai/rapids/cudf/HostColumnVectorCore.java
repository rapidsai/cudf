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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * A class that holds Host side Column Vector APIs and the OffHeapState.
 * Any children of a HostColumnVector will be instantiated via this class.
 */
public class HostColumnVectorCore implements AutoCloseable {

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
   * Returns the number of rows for a given host side column vector
   */
  public long getRowCount() {
    return rows;
  }

  /**
   * Return the element at a given row for a give data type
   * @param rowIndex the row number
   * @return an object that would need to be casted to appropriate type based on this vector's data type
   */
  Object getElement(int rowIndex) {
    if (type == DType.LIST) {
      List retList = new ArrayList();
      int start = offHeap.offsets.getInt(rowIndex * DType.INT32.getSizeInBytes());
      int end = offHeap.offsets.getInt((rowIndex + 1) * DType.INT32.getSizeInBytes());
      for (int j = start; j < end; j++) {
        retList.add(children.get(0).getElement(j));
      }
      return retList;
    } else if (type == DType.STRING) {
      int start = offHeap.offsets.getInt(rowIndex * DType.INT32.getSizeInBytes());
      int end = offHeap.offsets.getInt((rowIndex + 1) * DType.INT32.getSizeInBytes());
      int size = end - start;
      byte[] rawData = new byte[size];
      if (size > 0) {
        offHeap.data.getBytes(rawData, 0, start, size);
        return new String(rawData);
      } else if (isNull(rowIndex)) {
        return null;
      } else {
        return new String();
      }
    } else if (type == DType.STRUCT) {
      HostColumnVector.ColumnBuilder.StructType structType = new HostColumnVector.ColumnBuilder.StructType(true, 2);
      structType.addChild(new HostColumnVector.ColumnBuilder.BasicType(true, 2, DType.STRING));
      structType.addChild(new HostColumnVector.ColumnBuilder.BasicType(true, 2, DType.STRING));
      return getStruct(rowIndex,structType);
    } else {
      if (isNull(rowIndex)) {
        return null;
      }
      int start = rowIndex * type.getSizeInBytes();
      return readValue(start);
    }
  }

  HostColumnVector.ColumnBuilder.StructData getStruct(int rowIndex, HostColumnVector.ColumnBuilder.DataType mainType) {
    assert rowIndex < rows;
    assert type == DType.STRUCT;
    List<Object> retList = new ArrayList<>();
    // check if null or empty
    if (isNull(rowIndex)) {
      return null;
    }
    int numChildren = mainType.getNumChildren();
    for (int k = 0; k < numChildren; k++) {
      retList.add(children.get(k).getElement(rowIndex));
    }
    return new HostColumnVector.ColumnBuilder.StructData(retList);
  }
  /**
   * Method that returns a boolean to indicate if the element at a given row index is null
   * @param rowIndex the row index
   * @return true if null else false
   */
  public boolean isNull(long rowIndex) {
    assert (rowIndex >= 0 && rowIndex < rows) : "index is out of range 0 <= " + rowIndex + " < " + rows;
    if (offHeap.valid != null) {
      return BitVectorHelper.isNull(offHeap.valid, rowIndex);
    }
    return false;
  }

  /**
   * Helper method that reads in a value at a given row index
   * @param rowIndex the row index
   * @return an object that would need to be casted to appropriate type based on this vector's data type
   */
  private Object readValue(int rowIndex) {
//    assert rowIndex < rows * type.getSizeInBytes() : "rowIndex=" + rowIndex + " rhs=" + rows * type.getSizeInBytes();
    switch (type) {
      case INT32: // fall through
      case UINT32: // fall through
      case TIMESTAMP_DAYS:
      case DURATION_DAYS: return offHeap.data.getInt(rowIndex);
      case INT64: // fall through
      case UINT64: // fall through
      case DURATION_MICROSECONDS: // fall through
      case DURATION_MILLISECONDS: // fall through
      case DURATION_NANOSECONDS: // fall through
      case DURATION_SECONDS: // fall through
      case TIMESTAMP_MICROSECONDS: // fall through
      case TIMESTAMP_MILLISECONDS: // fall through
      case TIMESTAMP_NANOSECONDS: // fall through
      case TIMESTAMP_SECONDS: return offHeap.data.getLong(rowIndex);
      case FLOAT32: return offHeap.data.getFloat(rowIndex);
      case FLOAT64: return offHeap.data.getDouble(rowIndex);
      case UINT8: // fall through
      case INT8: return offHeap.data.getByte(rowIndex);
      case UINT16: // fall through
      case INT16: return offHeap.data.getShort(rowIndex);
      case BOOL8: return offHeap.data.getBoolean(rowIndex);
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
