/*
 *
 *  Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * This class represents settings for writing Parquet files. It includes meta data information
 * that will be used by the Parquet writer to write the file
 */
public final class ParquetWriterOptions extends ParquetColumnWriterOptions.ParquetStructColumnWriterOptions {
  private final CompressionType compressionType;
  private final Map<String, String> metadata;
  private final StatisticsFrequency statsGranularity;

  private ParquetWriterOptions(Builder builder) {
    super(builder);
    this.statsGranularity = builder.statsGranularity;
    this.compressionType = builder.compressionType;
    this.metadata = builder.metadata;
  }

  @Override
  boolean[] getFlatIsTimeTypeInt96() {
    return super.getFlatBooleans(new boolean[]{}, (opt) -> opt.getFlatIsTimeTypeInt96());
  }

  @Override
  int[] getFlatPrecision() {
    return super.getFlatInts(new int[]{}, (opt) -> opt.getFlatPrecision());
  }

  @Override
  int[] getFlatNumChildren() {
    return super.getFlatInts(new int[]{}, (opt) -> opt.getFlatNumChildren());
  }

  @Override
  boolean[] getFlatIsNullable() {
    return super.getFlatBooleans(new boolean[]{}, (opt) -> opt.getFlatIsNullable());
  }

  @Override
  boolean[] getFlatIsMap() {
    return super.getFlatBooleans(new boolean[]{}, (opt) -> opt.getFlatIsMap());
  }

  @Override
  String[] getFlatColumnNames() {
    return super.getFlatColumnNames(new String[]{});
  }

  String[] getMetadataKeys() {
    return metadata.keySet().toArray(new String[metadata.size()]);
  }

  String[] getMetadataValues() {
    return metadata.values().toArray(new String[metadata.size()]);
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

  public StatisticsFrequency getStatisticsFrequency() {
    return statsGranularity;
  }

  public CompressionType getCompressionType() {
    return compressionType;
  }

  public Map<String, String> getMetadata() {
    return metadata;
  }

  public int getTopLevelChildren() {
    return childColumnOptions.length;
  }

  public static class Builder extends ParquetColumnWriterOptions.AbstractStructBuilder<Builder,
      ParquetWriterOptions> {
    private StatisticsFrequency statsGranularity = StatisticsFrequency.ROWGROUP;
    final Map<String, String> metadata = new LinkedHashMap<>();
    CompressionType compressionType = CompressionType.AUTO;

    public Builder() {
      super();
    }

    /**
     * Add a metadata key and a value
     */
    public Builder withMetadata(String key, String value) {
      this.metadata.put(key, value);
      return this;
    }

    /**
     * Add a map of metadata keys and values
     */
    public Builder withMetadata(Map<String, String> metadata) {
      this.metadata.putAll(metadata);
      return this;
    }

    /**
     * Set the compression type to use for writing
     */
    public Builder withCompressionType(CompressionType compression) {
      this.compressionType = compression;
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
