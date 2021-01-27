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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;

/**
 * Column builder from Arrow data. This builder takes in pointers to the Arrow off heap
 * memory and allows efficient building of CUDF ColumnVectors from that arrow data.
 * The caller can add multiple batches where each batch corresponds to Arrow data
 * and those batches get concatenated together after being converted to CUDF
 * ColumnVectors.
 */
public final class ArrowColumnBuilder implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(ArrowColumnBuilder.class);

    private DType type;
    private ArrayList<Long> data = new ArrayList<>();
    private ArrayList<Long> dataLength = new ArrayList<>();
    private ArrayList<Long> validity = new ArrayList<>();
    private ArrayList<Long> validityLength = new ArrayList<>();
    private ArrayList<Long> offsets = new ArrayList<>();
    private ArrayList<Long> offsetsLength = new ArrayList<>();
    private ArrayList<Long> nullCount = new ArrayList<>();
    private ArrayList<Long> rows = new ArrayList<>();
    private int numBatches = 0;
    private String colName;

    public ArrowColumnBuilder(HostColumnVector.DataType type, String name) {
      this.type = type.getType();
      this.colName = name;
    }

    public void addBatch(long rows, long nullCount, long data, long dataLength, long valid,
                         long validLength, long offsets, long offsetsLength) {
      this.numBatches += 1;
      this.rows.add(rows);
      this.nullCount.add(nullCount);
      this.data.add(data);
      this.dataLength.add(dataLength);
      this.validity.add(valid);
      this.validityLength.add(validLength);
      this.offsets.add(offsets);
      this.offsetsLength.add(offsetsLength);
    }

    /**
     * Create the immutable ColumnVector, copied to the device based on the Arrow data.
     */
    public final ColumnVector buildAndPutOnDevice() {
      log.warn("buildAndPutOnDevice type before is: " + type + " name is: " + colName + " num batches is: " + this.numBatches);
      ArrayList<ColumnVector> allVecs = new ArrayList<>();
      for (int i = 0; i < this.numBatches; i++) {
        allVecs.add(ColumnVector.fromArrow(type, colName, rows.get(i), nullCount.get(i),
          data.get(i), dataLength.get(i), validity.get(i), validityLength.get(i),
          offsets.get(i), offsetsLength.get(i)));
      }
      ColumnVector vecRet;
      if (this.numBatches == 1) {
        vecRet = allVecs.get(0);
      } else if (this.numBatches > 1) {
        vecRet = ColumnVector.concatenate(allVecs.toArray(new ColumnVector[0]));
      } else {
        throw new IllegalStateException("Can't build a ColumnVector when no Arrow batches specified");
      }
      return vecRet;
    }

    @Override
    public void close() {
      // memory buffers owned outside of this
    }

    @Override
    public String toString() {
      StringJoiner sj = new StringJoiner(",");
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
