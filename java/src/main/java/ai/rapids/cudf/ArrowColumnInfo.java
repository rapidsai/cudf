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
 * Column information from Arrow data. 
 * This currently only supports primitive types and Strings, Decimals and nested types
 * such as list and struct are not supported.
 * The caller is responsible for eventually freeing the underlying Arrow array.
 */
public final class ArrowColumnInfo {
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

  public long getArrowArrayHandle() {
    return arrowArrayHandle;
  }

  public long getDataBufferAddress() {
    return dataBufferAddr;
  }

  public long getDataBufferSize() {
    return dataBufferSize;
  }

  public long getValidityBufferAddress() {
    return validityBufferAddr;
  }

  public long getValidityBufferSize() {
    return validityBufferSize;
  }

  public long getOffsetsBufferAddress() {
    return offsetsBufferAddr;
  }

  public long getOffsetsBufferSize() {
    return offsetsBufferSize;
  }

  public long getNumRows() {
    return numRows;
  }

  public long getNullCount() {
    return nullCount;
  }
}
