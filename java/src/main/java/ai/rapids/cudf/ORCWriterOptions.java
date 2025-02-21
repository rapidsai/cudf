/*
 *
 *  Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
 * This class represents settings for writing ORC files. It includes meta data information
 * that will be used by the ORC writer to write the file.
 */
public class ORCWriterOptions extends CompressionMetadataWriterOptions {
  private int stripeSizeRows;

  private ORCWriterOptions(Builder builder) {
    super(builder);
    this.stripeSizeRows = builder.stripeSizeRows;
  }

  public static Builder builder() {
    return new Builder();
  }

  public int getStripeSizeRows() {
    return stripeSizeRows;
  }

  public static class Builder extends CompressionMetadataWriterOptions.Builder
          <Builder, ORCWriterOptions> {
    // < 1M rows default orc stripe rows, defined in cudf/cpp/include/cudf/io/orc.hpp
    private int stripeSizeRows = 1000000;

    public Builder withStripeSizeRows(int stripeSizeRows) {
      // maximum stripe size cannot be smaller than 512
      if (stripeSizeRows < 512) {
        throw new IllegalArgumentException("Maximum stripe size cannot be smaller than 512");
      }
      this.stripeSizeRows = stripeSizeRows;
      return this;
    }

    public ORCWriterOptions build() {
      return new ORCWriterOptions(this);
    }
  }
}
