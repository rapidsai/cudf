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

import java.util.ArrayList;

/**
 * Column builder from Arrow data. This builder takes in pointers to the Arrow off heap
 * memory and allows efficient building of CUDF ColumnVectors from that Arrow data.
 * The caller can add multiple batches where each batch corresponds to Arrow data
 * and those batches get concatenated together after being converted to CUDF
 * ColumnVectors.
 * This currently only supports primitive types and Strings, Decimals and nested types
 * such as list and struct are not supported.
 */
public final class ArrowColumnBuilder implements AutoCloseable {
    private DType type;
    private final ArrayList<Long> data = new ArrayList<>();
    private final ArrayList<Long> dataLength = new ArrayList<>();
    private final ArrayList<Long> validity = new ArrayList<>();
    private final ArrayList<Long> validityLength = new ArrayList<>();
    private final ArrayList<Long> offsets = new ArrayList<>();
    private final ArrayList<Long> offsetsLength = new ArrayList<>();
    private final ArrayList<Long> nullCount = new ArrayList<>();
    private final ArrayList<Long> rows = new ArrayList<>();

    public ArrowColumnBuilder(HostColumnVector.DataType type) {
      this.type = type.getType();
    }

    /**
     * Add an Arrow buffer. This api allows you to add multiple if you want them
     * combined into a single ColumnVector.
     * Note, this takes all data, validity, and offsets buffers, but they may not all
     * be needed based on the data type. The buffer and length should be set to 0
     * if they aren't used for that type.
     * This api only supports primitive types and Strings, Decimals and nested types
     * such as list and struct are not supported.
     * @param rows - number of rows in this Arrow buffer
     * @param nullCount - number of null values in this Arrow buffer
     * @param data - memory address of the Arrow data buffer
     * @param dataLength - size of the Arrow data buffer in bytes
     * @param validity - memory address of the Arrow validity buffer
     * @param validLength - size of the Arrow validity buffer in bytes
     * @param offsets - memory address of the Arrow offsets buffer
     * @param offsetsLenght - size of the Arrow offsets buffer in bytes
     */
    public void addBatch(long rows, long nullCount, long data, long dataLength, long validity,
                         long validityLength, long offsets, long offsetsLength) {
      this.rows.add(rows);
      this.nullCount.add(nullCount);
      this.data.add(data);
      this.dataLength.add(dataLength);
      this.validity.add(validity);
      this.validityLength.add(validityLength);
      this.offsets.add(offsets);
      this.offsetsLength.add(offsetsLength);
    }

    /**
     * Create the immutable ColumnVector, copied to the device based on the Arrow data.
     * @return - new ColumnVector
     */
    public final ColumnVector buildAndPutOnDevice() {
      int numBatches = rows.size();
      ArrayList<ColumnVector> allVecs = new ArrayList<>(numBatches);
      ColumnVector vecRet;
      try {
        for (int i = 0; i < numBatches; i++) {
          allVecs.add(ColumnVector.fromArrow(type, rows.get(i), nullCount.get(i),
          data.get(i), dataLength.get(i), validity.get(i), validityLength.get(i),
          offsets.get(i), offsetsLength.get(i)));
        }
        if (numBatches == 1) {
          vecRet = allVecs.get(0);
        } else if (numBatches > 1) {
          vecRet = ColumnVector.concatenate(allVecs.toArray(new ColumnVector[0]));
        } else {
          throw new IllegalStateException("Can't build a ColumnVector when no Arrow batches specified");
        }
      } finally {
        // close the vectors that were concatenated
        if (numBatches > 1) {
          for (ColumnVector cv : allVecs) {
            cv.close();
          }
        }
      }
      return vecRet;
    }

    @Override
    public void close() {
      // memory buffers owned outside of this
    }

    @Override
    public String toString() {
      return "ArrowColumnBuilder{" +
          "type=" + type +
          ", data=" + data +
          ", dataLength=" + dataLength +
          ", validity=" + validity +
          ", validityLength=" + validityLength +
          ", offsets=" + offsets +
          ", offsetsLength=" + offsetsLength+
          ", nullCount=" + nullCount +
          ", rows=" + rows +
          ", populatedRows=" + rows +
          '}';
    }
}
