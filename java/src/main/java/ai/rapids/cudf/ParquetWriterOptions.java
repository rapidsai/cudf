/*
 *
 *  Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
 * This class represents settings for writing Parquet files. It includes meta data information
 * that will be used by the Parquet writer to write the file
 */
public final class ParquetWriterOptions extends CompressionMetadataWriterOptions {
  private final StatisticsFrequency statsGranularity;
  private int rowGroupSizeRows;
  private long rowGroupSizeBytes;

  private ParquetWriterOptions(Builder builder) {
    super(builder);
    this.rowGroupSizeRows = builder.rowGroupSizeRows;
    this.rowGroupSizeBytes = builder.rowGroupSizeBytes;
    this.statsGranularity = builder.statsGranularity;
  }

  public enum StatisticsFrequency {
    /** Do not generate statistics */
    NONE(0),

    /** Generate column statistics for each rowgroup */
    ROWGROUP(1),

    /** Generate column statistics for each page */
    PAGE(2);

    final int nativeId;

    StatisticsFrequency(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  public int getRowGroupSizeRows() {
    return rowGroupSizeRows;
  }

  public long getRowGroupSizeBytes() {
    return rowGroupSizeBytes;
  }

  public StatisticsFrequency getStatisticsFrequency() {
    return statsGranularity;
  }

  public static class Builder extends CompressionMetadataWriterOptions.Builder
        <Builder, ParquetWriterOptions> {
    private int rowGroupSizeRows = 1000000; //Max of 1 million rows per row group
    private long rowGroupSizeBytes = 128 * 1024 * 1024; //Max of 128MB per row group
    private StatisticsFrequency statsGranularity = StatisticsFrequency.ROWGROUP;

    public Builder() {
      super();
    }

    public Builder withRowGroupSizeRows(int rowGroupSizeRows) {
      this.rowGroupSizeRows = rowGroupSizeRows;
      return this;
    }

    public Builder withRowGroupSizeBytes(long rowGroupSizeBytes) {
      this.rowGroupSizeBytes = rowGroupSizeBytes;
      return this;
    }

    public Builder withStatisticsFrequency(StatisticsFrequency statsGranularity) {
      this.statsGranularity = statsGranularity;
      return this;
    }

    public ParquetWriterOptions build() {
      return new ParquetWriterOptions(this);
    }
  }
}
