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

import java.util.Arrays;

/**
 * A table that is backed by a single contiguous device buffer. This makes transfers of the data
 * much simpler.
 */
public final class ContiguousTable implements AutoCloseable {
  private Table table;
  private DeviceMemoryBuffer buffer;

  //Will be called from JNI
  static ContiguousTable fromContiguousColumnViews(long[] columnViewAddresses,
                                           long address, long lengthInBytes, long rmmBufferAddress) {
    Table table = null;
    ColumnVector[] vectors = new ColumnVector[columnViewAddresses.length];
    DeviceMemoryBuffer buffer = DeviceMemoryBuffer.fromRmm(address, lengthInBytes, rmmBufferAddress);
    try {
      for (int i = 0; i < vectors.length; i++) {
        vectors[i] = ColumnVector.fromViewWithContiguousAllocation(columnViewAddresses[i], buffer);
      }
      table = new Table(vectors);
      ContiguousTable ret = new ContiguousTable(table, buffer);
      buffer = null;
      table = null;
      return ret;
    } finally {
      if (buffer != null) {
        buffer.close();
      }

      for (int i = 0; i < vectors.length; i++) {
        if (vectors[i] != null) {
          vectors[i].close();
        }
      }

      if (table != null) {
        table.close();
      }
    }
  }

  private ContiguousTable(Table table, DeviceMemoryBuffer buffer) {
    this.table = table;
    this.buffer = buffer;
  }

  public Table getTable() {
    return table;
  }

  public DeviceMemoryBuffer getBuffer() {
    return buffer;
  }

  @Override
  public void close() {
    if (table != null) {
      table.close();
      table = null;
    }

    if (buffer != null) {
      buffer.close();
      buffer = null;
    }
  }
}