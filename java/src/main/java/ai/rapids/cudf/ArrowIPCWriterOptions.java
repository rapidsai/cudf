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

  private ArrowIPCWriterOptions(Builder builder) {
    super(builder);
    this.size = builder.size;
    this.callback = builder.callback;
  }

  public long getMaxChunkSize() {
    return size;
  }

  public DoneOnGpu getCallback() {
    return callback;
  }

  public static class Builder extends WriterBuilder<Builder> {
    private long size = -1;
    private DoneOnGpu callback = (ignored) -> {};

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

    public ArrowIPCWriterOptions build() {
      return new ArrowIPCWriterOptions(this);
    }
  }

  public static final ArrowIPCWriterOptions DEFAULT = new ArrowIPCWriterOptions(new Builder());

  public static Builder builder() {
    return new Builder();
  }
}
