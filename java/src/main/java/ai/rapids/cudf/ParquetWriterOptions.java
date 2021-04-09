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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * This class represents settings for writing Parquet files. It includes meta data information
 * that will be used by the Parquet writer to write the file
 */
public class ParquetWriterOptions {
  private final CompressionType compressionType;
  private final Map<String, String> metadata;
  private final StatisticsFrequency statsGranularity;
  private final List<ParquetColumnWriterOptions> columnOptions;

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

  public CompressionType getCompressionType() {
    return compressionType;
  }

  public Map<String, String> getMetadata() {
    return metadata;
  }

  String[] getMetadataKeys() {
    return metadata.keySet().toArray(new String[metadata.size()]);
  }

  String[] getMetadataValues() {
    return metadata.values().toArray(new String[metadata.size()]);
  }

  public static class Builder {
    final Map<String, String> metadata = new LinkedHashMap<>();
    CompressionType compressionType = CompressionType.AUTO;
    private StatisticsFrequency statsGranularity = StatisticsFrequency.ROWGROUP;
    private List<ParquetColumnWriterOptions> columnOptions = new ArrayList();

    /**
     * Add a metadata key and a value
     * @param key
     * @param value
     */
    public Builder withMetadata(String key, String value) {
      this.metadata.put(key, value);
      return this;
    }

    /**
     * Add a map of metadata keys and values
     * @param metadata
     */
    public Builder withMetadata(Map<String, String> metadata) {
      this.metadata.putAll(metadata);
      return this;
    }

    /**
     * Set the compression type to use for writing
     * @param compression
     */
    public Builder withCompressionType(CompressionType compression) {
      this.compressionType = compression;
      return this;
    }

    public Builder withStatisticsFrequency(StatisticsFrequency statsGranularity) {
      this.statsGranularity = statsGranularity;
      return this;
    }

    /**
     * Set column name
     * @param name
     */
    public Builder withColumn(boolean nullable, String... name) {
      IntStream.range(0, name.length).forEach(
          i -> columnOptions.add(ParquetColumnWriterOptions.leafBuilder(name[i])
              .withNullable(nullable)
              .build())
      );
      return this;
    }

    /**
     * Set column name
     * @param name
     */
    public Builder withColumn(String... name) {
      withColumn(false, name);
      return this;
    }

    /**
     * Set nullable column meta data
     * @param name
     */
    public Builder withNullableColumn(String... name) {
      withColumn(true, name);
      return this;
    }

    /**
     * Set decimal column meta data
     * @param name
     * @param precision
     */
    public Builder withDecimalColumn(String name, int precision, boolean nullable) {
      columnOptions.add(ParquetColumnWriterOptions.leafBuilder(name)
          .withDecimalPrecision(precision)
          .withNullable(nullable)
          .build());
      return this;
    }

    /**
     * Set decimal column meta data
     * @param name
     * @param precision
     */
    public Builder withDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, false);
      return this;
    }

    /**
     * Set nullable decimal column meta data
     * @param name
     * @param precision
     */
    public Builder withNullableDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, true);
      return this;
    }

    /**
     * Create a timestamp column meta data
     * @param name
     * @param isInt96
     */
    public Builder withTimestampColumn(String name, boolean isInt96, boolean nullable) {
      columnOptions.add(ParquetColumnWriterOptions.leafBuilder(name)
          .withTimestampInt96(isInt96)
          .withNullable(nullable)
          .build());
      return this;
    }

    /**
     * Create a timestamp column meta data
     * @param name
     * @param isInt96
     */
    public Builder withTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, false);
      return this;
    }

    /**
     * Create a nullable timestamp column meta data
     * @param name
     * @param isInt96
     */
    public Builder withNullableTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, true);
      return this;
    }

    /**
     * Set a struct column with these options
     * @param option
     */
    public Builder withStructColumn(ParquetColumnWriterOptions.ParquetStructColumnWriterOptions option) {
      for (ParquetColumnWriterOptions opt: option.getChildColumnOptions()) {
        if (opt.getColumName().isEmpty()) {
          throw new IllegalArgumentException("Column name can't be empty");
        }
      }
      columnOptions.add(option);
      return this;
    }

    /**
     * Set a column with these options
     * @param options
     */
    public Builder withListColumn(ParquetColumnWriterOptions.ParquetListColumnWriterOptions options) {
      // Lists should have only one child in the Java bindings, unfortunately we have to do this
      // because the way cudf is implemented today, it requires two children and then it drops
      // the first one assuming its the offsets child
      assert (options.getChildColumnOptions().length == 2) : "Lists can only have two children";
      if (options.getChildColumnOptions()[0] != ParquetColumnWriterOptions.DUMMY_CHILD) {
        throw new IllegalArgumentException("First child in the list has to be DUMMY_CHILD");
      }
      if (options.getChildColumnOptions()[1].getColumName().isEmpty()) {
        throw new IllegalArgumentException("Column name can't be empty");
      }
      columnOptions.add(options);
      return this;
    }

    public ParquetWriterOptions build() {
      return new ParquetWriterOptions(this);
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  private ParquetWriterOptions(Builder builder) {
    this.statsGranularity = builder.statsGranularity;
    this.columnOptions = builder.columnOptions;
    this.compressionType = builder.compressionType;
    this.metadata = builder.metadata;
  }

  public StatisticsFrequency getStatisticsFrequency() {
    return statsGranularity;
  }

  boolean[] getFlatIsTimeTypeInt96() {
    List<Boolean> a = new ArrayList<>();
    a.add(false); // dummy value
    for (ParquetColumnWriterOptions opt: columnOptions) {
      a.addAll(opt.getFlatIsTimeTypeInt96());
    }
    final boolean[] primitivesBool = new boolean[a.size()];
    int i = 0;
    for (Boolean b: a) {
      primitivesBool[i++] = b;
    }
    return primitivesBool;
  }

  int[] getFlatPrecision() {
    List<Integer> a = new ArrayList<>();
    a.add(0); // dummy value
    for (ParquetColumnWriterOptions opt: columnOptions) {
      a.addAll(opt.getFlatPrecision());
    }
    return a.stream().mapToInt(Integer::intValue).toArray();
  }

  boolean[] getFlatIsNullable() {
    List<Boolean> a = new ArrayList<>();
    a.add(false); // dummy value
    for (ParquetColumnWriterOptions opt: columnOptions) {
      a.addAll(opt.getFlatIsNullable());
    }
    final boolean[] primitivesBool = new boolean[a.size()];
    int i = 0;
    for (Boolean b: a) {
      primitivesBool[i++] = b;
    }
    return primitivesBool;
  }

  int[] getFlatNumChildren() {
    List<Integer> a = new ArrayList<>();
    a.add(columnOptions.size());
    for (ParquetColumnWriterOptions opt: columnOptions) {
      a.addAll(opt.getFlatNumChildren());
    }
    return a.stream().mapToInt(Integer::intValue).toArray();
  }

  String[] getFlatColumnNames() {
    List<String> a = new ArrayList<>();
    a.add(""); // dummy value to keep the code simple
    for (ParquetColumnWriterOptions opt: columnOptions) {
      a.addAll(opt.getFlatColumnNames());
    }
    return a.stream().toArray(String[]::new);
  }
}
