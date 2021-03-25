/*
 *
 *  Copyright (c) 2021, NVIDIA CORPORATION.
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

import java.nio.ByteBuffer;
import java.util.ArrayList;

/**
 * Holds information about the underlying Arrow column. This returns addresses and sizes
 * of the underlying Arrow buffers that can be used to reconstruct the Arrow vector.
 *
 * This currently only supports primitive types and Strings, Decimals and nested types
 * such as list and struct are not supported. DURATION_DAYS is also not supported by cudf.
 *
 * The caller is responsible for eventually freeing the underlying Arrow array by 
 * calling close() when they are finished with the Arrow vector.
 */
public final class ArrowColumnInfo implements AutoCloseable {
  private long arrowArrayHandle;
  private long dataBufferAddr;
  private long dataBufferSize;
  private long numRows;
  private long validityBufferAddr;
  private long validityBufferSize;
  private long nullCount;
  private long offsetsBufferAddr;
  private long offsetsBufferSize;

  public ArrowColumnInfo(long arrowArrayHandle, long dataAddr, long dataSize, long rows,
      long validityAddr, long validitySize, long nullCount) {
    this.arrowArrayHandle = arrowArrayHandle;
    this.dataBufferAddr = dataAddr;
    this.dataBufferSize = dataSize;
    this.numRows = rows;
    this.validityBufferAddr = validityAddr;
    this.validityBufferSize = validitySize;
    this.nullCount = nullCount;
    this.offsetsBufferAddr = 0;
    this.offsetsBufferSize = 0;
  }

  public ArrowColumnInfo(long arrowArrayHandle, long dataAddr, long dataSize, long rows,
      long validityAddr, long validitySize, long nullCount, long offsetsAddr, long offsetsSize) {
    this(arrowArrayHandle, dataAddr, dataSize, rows, validityAddr, validitySize, nullCount);
    this.offsetsBufferAddr = offsetsAddr;
    this.offsetsBufferSize = offsetsSize;
  }

  @Override
  public void close() {
    ColumnVector.closeArrowArray(arrowArrayHandle);
  }

  /**
   * Get the Arrow data buffer address.
   * @return arrow data buffer address
   */
  public long getDataBufferAddress() {
    return dataBufferAddr;
  }

  /**
   * Get the Arrow data buffer size.
   * @return arrow data buffer size 
   */
  public long getDataBufferSize() {
    return dataBufferSize;
  }

  /**
   * Get the Arrow validity buffer address.
   * @return arrow validity buffer address
   */
  public long getValidityBufferAddress() {
    return validityBufferAddr;
  }

  /**
   * Get the Arrow validity buffer size.
   * @return arrow validity buffer size
   */
  public long getValidityBufferSize() {
    return validityBufferSize;
  }

  /**
   * Get the Arrow offsets buffer address.
   * @return arrow offsets buffer address
   */
  public long getOffsetsBufferAddress() {
    return offsetsBufferAddr;
  }

  /**
   * Get the Arrow offsets buffer size.
   * @return arrow offsets buffer size
   */
  public long getOffsetsBufferSize() {
    return offsetsBufferSize;
  }

  /**
   * Get the number of rows in the Arrow column
   * @return number of rows
   */
  public long getNumRows() {
    return numRows;
  }

  /**
   * Get the Arrow null count
   * @return null count
   */
  public long getNullCount() {
    return nullCount;
  }
}
