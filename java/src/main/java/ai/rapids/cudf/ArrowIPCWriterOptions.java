/*
 *
 *  Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
import java.util.Arrays;
import java.util.List;

/**
 * Settings for writing Arrow IPC data.
 */
public class ArrowIPCWriterOptions extends WriterOptions {

  public interface DoneOnGpu {
    /**
     * A callback to indicate that the table is off of the GPU
     * and may be closed, even if all of the data is not yet written.
     * @param table the table that can be closed.
     */
    void doneWithTheGpu(Table table);
  }

  private final long size;
  private final DoneOnGpu callback;
  private final ColumnMetadata[] columnMeta;

  private ArrowIPCWriterOptions(Builder builder) {
    super(builder);
    this.size = builder.size;
    this.callback = builder.callback;
    this.columnMeta = builder.columnMeta.toArray(new ColumnMetadata[builder.columnMeta.size()]);
  }

  public long getMaxChunkSize() {
    return size;
  }

  public DoneOnGpu getCallback() {
    return callback;
  }

  public ColumnMetadata[] getColumnMetadata() {
    if (columnMeta == null || columnMeta.length == 0) {
      // For compatibility. Try building from column names when column meta is empty.
      // Should remove this once all the callers update to use only column metadata.
      return Arrays
              .stream(getColumnNames())
              .map(ColumnMetadata::new)
              .toArray(ColumnMetadata[]::new);
    } else {
      return columnMeta;
    }
  }

  public static class Builder extends WriterBuilder<Builder> {
    private long size = -1;
    private DoneOnGpu callback = (ignored) -> {};
    private List<ColumnMetadata> columnMeta = new ArrayList<>();

    public Builder withMaxChunkSize(long size) {
      this.size = size;
      return this;
    }

    public Builder withCallback(DoneOnGpu callback) {
      if (callback == null) {
        this.callback = (ignored) -> {};
      } else {
        this.callback = callback;
      }
      return this;
    }

    /**
     * This should be used instead of `withColumnNames` when there are children
     * columns of struct type.
     */
    public Builder withColumnMetadata(ColumnMetadata... columnMeta) {
      this.columnMeta.addAll(Arrays.asList(columnMeta));
      return this;
    }

    public ArrowIPCWriterOptions build() {
      return new ArrowIPCWriterOptions(this);
    }
  }

  public static final ArrowIPCWriterOptions DEFAULT = new ArrowIPCWriterOptions(new Builder());

  public static Builder builder() {
    return new Builder();
  }
}
