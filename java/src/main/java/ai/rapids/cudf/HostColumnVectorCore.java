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

public class BaseHostColumnVector implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(HostColumnVector.class);

  protected final OffHeapState offHeap;
  protected final DType type;
  protected long rows;
  protected Optional<Long> nullCount = Optional.empty();
  protected List<BaseHostColumnVector> children = new ArrayList<>();

  public BaseHostColumnVector(DType type, long rows,
                              Optional<Long> nullCount, HostMemoryBuffer data, HostMemoryBuffer validity,
                              HostMemoryBuffer offsets, List<BaseHostColumnVector> nestedChildren) {
    this.offHeap = new OffHeapState(data, validity,  offsets);
    MemoryCleaner.register(this, offHeap);
    this.type = type;
    this.rows = rows;
    this.nullCount = nullCount;
    this.children = nestedChildren;
  }

  @Override
  public void close() {
    for (BaseHostColumnVector child : children) {
      if (child != null) {
        child.close();
      }
    }
    offHeap.delRef();
    offHeap.cleanImpl(false);
  }

  HostMemoryBuffer getData() {
    return offHeap.data;
  }

  HostMemoryBuffer getValidity() {
    return offHeap.valid;
  }

  DType getType() {
    return type;
  }

  long getNullCount() {
    return nullCount.get();
  }

  List<BaseHostColumnVector> getNestedChildren() {
    return children;
  }

  public HostMemoryBuffer getOffsets() {
    return offHeap.offsets;
  }

  public long getRows() {
    return rows;
  }


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
    if (offHeap.valid != null) {
      return BitVectorHelper.isNull(offHeap.valid, index);
    }
    return false;
  }

  private Object readValue(int index){
    assert index < rows * type.getSizeInBytes();
    switch (type) {
      case INT32: // fall through
      case UINT32: // fall through
      case TIMESTAMP_DAYS:
      case DURATION_DAYS: return offHeap.data.getInt(index);
      case INT64: // fall through
      case UINT64: // fall through
      case DURATION_MICROSECONDS: // fall through
      case DURATION_MILLISECONDS: // fall through
      case DURATION_NANOSECONDS: // fall through
      case DURATION_SECONDS: // fall through
      case TIMESTAMP_MICROSECONDS: // fall through
      case TIMESTAMP_MILLISECONDS: // fall through
      case TIMESTAMP_NANOSECONDS: // fall through
      case TIMESTAMP_SECONDS: return offHeap.data.getLong(index);
      case FLOAT32: return offHeap.data.getFloat(index);
      case FLOAT64: return offHeap.data.getDouble(index);
      case UINT8: // fall through
      case INT8: return offHeap.data.getByte(index);
      case UINT16: // fall through
      case INT16: return offHeap.data.getShort(index);
      case BOOL8: return offHeap.data.getBoolean(index);
      default: throw new UnsupportedOperationException("Do not support " + type);
    }
  }

  public long getHostMemorySize() {
    long totalSize = 0;
    if (offHeap.data != null) {
      totalSize += offHeap.data.length;
    }
    if (offHeap.offsets != null) {
      totalSize += offHeap.offsets.length;
    }
    if (offHeap.valid != null) {
      totalSize += offHeap.valid.length;
    }
    for (BaseHostColumnVector nhcv : children) {
      totalSize += nhcv.getHostMemorySize();
    }
    return totalSize;
  }

  @Override
  public String toString() {
    return "BaseHostColumnVector{" +
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
