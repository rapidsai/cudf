/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
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
