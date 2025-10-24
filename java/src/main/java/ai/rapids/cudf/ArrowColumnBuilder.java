/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.nio.ByteBuffer;
import java.util.ArrayList;

/**
 * Column builder from Arrow data. This builder takes in byte buffers referencing
 * Arrow data and allows efficient building of CUDF ColumnVectors from that Arrow data.
 * The caller can add multiple batches where each batch corresponds to Arrow data
 * and those batches get concatenated together after being converted to CUDF
 * ColumnVectors.
 * This currently only supports primitive types and Strings, Decimals and nested types
 * such as list and struct are not supported.
 */
public final class ArrowColumnBuilder implements AutoCloseable {
    private DType type;
    private final ArrayList<ByteBuffer> data = new ArrayList<>();
    private final ArrayList<ByteBuffer> validity = new ArrayList<>();
    private final ArrayList<ByteBuffer> offsets = new ArrayList<>();
    private final ArrayList<Long> nullCount = new ArrayList<>();
    private final ArrayList<Long> rows = new ArrayList<>();

    public ArrowColumnBuilder(HostColumnVector.DataType type) {
      this.type = type.getType();
    }

    /**
     * Add an Arrow buffer. This API allows you to add multiple if you want them
     * combined into a single ColumnVector.
     * Note, this takes all data, validity, and offsets buffers, but they may not all
     * be needed based on the data type. The buffer should be null if its not used
     * for that type.
     * This API only supports primitive types and Strings, Decimals and nested types
     * such as list and struct are not supported.
     * @param rows - number of rows in this Arrow buffer
     * @param nullCount - number of null values in this Arrow buffer
     * @param data - ByteBuffer of the Arrow data buffer
     * @param validity - ByteBuffer of the Arrow validity buffer
     * @param offsets - ByteBuffer of the Arrow offsets buffer
     */
    public void addBatch(long rows, long nullCount, ByteBuffer data, ByteBuffer validity,
                         ByteBuffer offsets) {
      this.rows.add(rows);
      this.nullCount.add(nullCount);
      this.data.add(data);
      this.validity.add(validity);
      this.offsets.add(offsets);
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
            data.get(i), validity.get(i), offsets.get(i)));
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
          allVecs.forEach(cv -> cv.close());
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
        ", validity=" + validity +
        ", offsets=" + offsets +
        ", nullCount=" + nullCount +
        ", rows=" + rows +
        '}';
    }
}
