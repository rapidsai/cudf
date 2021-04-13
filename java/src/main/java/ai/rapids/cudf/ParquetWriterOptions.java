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

import java.util.*;

/**
 * This class represents settings for writing Parquet files. It includes meta data information
 * that will be used by the Parquet writer to write the file
 */
public class ParquetWriterOptions {
  // This child is needed as the first child of a List column meta due to how cudf has been
  // implemented. Cudf drops the first child from the meta if a column is a LIST. This is done
  // this way due to some complications in the parquet reader. There was change to fix this here:
  // https://github.com/rapidsai/cudf/pull/7461/commits/5ce33b40abb87cc7b76b5efeb0a3a0215f9ef6fb
  // but it was reverted later on here:
  // https://github.com/rapidsai/cudf/pull/7461/commits/f248eb7265de995a95f998d46d897fb0ae47f53e
  private static final ParquetWriterOptions DUMMY_CHILD = new ParquetWriterOptions("DUMMY", false);

  private CompressionType compressionType;
  private Map<String, String> metadata;
  private StatisticsFrequency statsGranularity;
  private boolean isTimestampTypeInt96;
  private int precision;
  private boolean isNullable;
  private String columName = "";
  protected List<ParquetWriterOptions> columnOptions = new ArrayList<>();

  private ParquetWriterOptions(String name, boolean isInt96, int precision, boolean nullable) {
    this.columName = name;
    this.precision = precision;
    this.isNullable = nullable;
    this.isTimestampTypeInt96 = isInt96;
  }

  private ParquetWriterOptions(String name, boolean nullable) {
    this(name, false, 0, nullable);
  }

  private ParquetWriterOptions(String name, boolean isInt96, boolean nullable) {
    this(name, isInt96, 0, nullable);
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

  public static class ParquetStructColumnWriterOptions extends ParquetWriterOptions {
    protected ParquetStructColumnWriterOptions(StructBuilder builder) {
      super(builder);
    }
  }

  public static class ParquetListColumnWriterOptions extends ParquetWriterOptions {
    protected ParquetListColumnWriterOptions(ListBuilder builder) {
      super(builder);
    }
  }

  public static class Builder<T extends Builder> {
    final Map<String, String> metadata = new LinkedHashMap<>();
    CompressionType compressionType = CompressionType.AUTO;
    protected String name;
    protected boolean isNullable = true;
    private StatisticsFrequency statsGranularity = StatisticsFrequency.ROWGROUP;
    protected List<ParquetWriterOptions> columnOptions = new ArrayList();

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
    public T withColumn(boolean nullable, String... name) {
      for (String n: name) {
        columnOptions.add(new ParquetWriterOptions(n, nullable));
      }
      return (T) this;
    }

    /**
     * Set column name
     * @param name
     */
    public T withNonNullableColumn(String... name) {
      withColumn(false, name);
      return (T) this;
    }

    /**
     * Set nullable column meta data
     * @param name
     */
    public T withNullableColumn(String... name) {
      withColumn(true, name);
      return (T) this;
    }

    /**
     * Set decimal column meta data
     * @param name
     * @param precision
     */
    public T withDecimalColumn(String name, int precision, boolean nullable) {
      columnOptions.add(new ParquetWriterOptions(name, false, precision, nullable));
      return (T) this;
    }

    /**
     * Set decimal column meta data
     * @param name
     * @param precision
     */
    public T withDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, false);
      return (T) this;
    }

    /**
     * Set nullable decimal column meta data
     * @param name
     * @param precision
     */
    public T withNullableDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, true);
      return (T) this;
    }

    /**
     * Create a timestamp column meta data
     * @param name
     * @param isInt96
     */
    public T withTimestampColumn(String name, boolean isInt96, boolean nullable) {
      columnOptions.add(new ParquetWriterOptions(name, isInt96, nullable));
      return (T) this;
    }

    /**
     * Create a timestamp column meta data
     * @param name
     * @param isInt96
     */
    public T withTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, false);
      return (T) this;
    }

    /**
     * Create a nullable timestamp column meta data
     * @param name
     * @param isInt96
     */
    public T withNullableTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, true);
      return (T) this;
    }

    /**
     * Set a struct column with these options
     * @param option
     */
    public T withStructColumn(ParquetStructColumnWriterOptions option) {
      for (ParquetWriterOptions opt: option.columnOptions) {
        if (opt.columName.isEmpty()) {
          throw new IllegalArgumentException("Column name can't be empty");
        }
      }
      columnOptions.add(option);
      return (T) this;
    }

    /**
     * Set a column with these options
     * @param options
     */
    public T withListColumn(ParquetListColumnWriterOptions options) {
      // Lists should have only one child in the Java bindings, unfortunately we have to do this
      // because the way cudf is implemented today, it requires two children and then it drops
      // the first one assuming its the offsets child
      assert (options.columnOptions.size() == 2) : "Lists can only have two children";
      if (options.columnOptions.get(0) != ParquetWriterOptions.DUMMY_CHILD) {
        throw new IllegalArgumentException("First child in the list has to be DUMMY_CHILD");
      }
      if (options.columnOptions.get(1).columName.isEmpty()) {
        throw new IllegalArgumentException("Column name can't be empty");
      }
      columnOptions.add(options);
      return (T) this;
    }

    public ParquetWriterOptions build() {
      return new ParquetWriterOptions(this);
    }
  }

  public static class StructBuilder extends Builder<StructBuilder> {

    public StructBuilder(String name, boolean isNullable) {
      this.name = name;
      this.isNullable = isNullable;
    }

    public ParquetStructColumnWriterOptions build() {
      return new ParquetStructColumnWriterOptions(this);
    }
  }

  public static class ListBuilder extends Builder<ListBuilder> {

    public ListBuilder(String name, boolean isNullable) {
      this.name = name;
      this.isNullable = isNullable;
    }

    public ParquetListColumnWriterOptions build() {
      assert(columnOptions.size() == 1) : "Lists can only have 1 child";
      columnOptions.add(0, DUMMY_CHILD);
      return new ParquetListColumnWriterOptions(this);
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  public static ListBuilder listBuilder(String name) {
    return new ListBuilder(name, true);
  }

  public static ListBuilder listBuilder(String name, boolean isNullable) {
    return new ListBuilder(name, isNullable);
  }

  public static StructBuilder structBuilder(String name, boolean isNullable) {
    return new StructBuilder(name, isNullable);
  }

  public static StructBuilder structBuilder(String name) {
    return new StructBuilder(name, true);
  }

  private ParquetWriterOptions(Builder builder) {
    this.statsGranularity = builder.statsGranularity;
    this.columnOptions = builder.columnOptions;
    this.compressionType = builder.compressionType;
    this.metadata = builder.metadata;
  }

  private ParquetWriterOptions(StructBuilder builder) {
    this.columName = builder.name;
    this.isNullable = builder.isNullable;
    this.columnOptions = builder.columnOptions;
  }

  private ParquetWriterOptions(ListBuilder builder) {
    assert (builder.columnOptions.size() == 2) : "Use the ListBuilder to add a list column";
    this.columName = builder.name;
    this.isNullable = builder.isNullable;
    this.columnOptions = builder.columnOptions;
  }

  public StatisticsFrequency getStatisticsFrequency() {
    return statsGranularity;
  }

  boolean[] getFlatIsTimeTypeInt96() {
    boolean[] ret = {isTimestampTypeInt96};

    for (ParquetWriterOptions opt: columnOptions) {
      boolean[] b = opt.getFlatIsTimeTypeInt96();
      boolean[] tmp = new boolean[ret.length + b.length];
      System.arraycopy(ret, 0, tmp, 0, ret.length);
      System.arraycopy(b, 0, tmp, ret.length, b.length);
      ret = tmp;
    }
    return ret;
  }

  int[] getFlatPrecision() {
    int[] ret = {precision};

    for (ParquetWriterOptions opt: columnOptions) {
      int[] b = opt.getFlatPrecision();
      int[] tmp = new int[ret.length + b.length];
      System.arraycopy(ret, 0, tmp, 0, ret.length);
      System.arraycopy(b, 0, tmp, ret.length, b.length);
      ret = tmp;
    }
    return ret;
  }

  boolean[] getFlatIsNullable() {
    boolean[] ret = {isNullable};

    for (ParquetWriterOptions opt: columnOptions) {
      boolean[] b = opt.getFlatIsNullable();
      boolean[] tmp = new boolean[ret.length + b.length];
      System.arraycopy(ret, 0, tmp, 0, ret.length);
      System.arraycopy(b, 0, tmp, ret.length, b.length);
      ret = tmp;
    }
    return ret;
  }

  int[] getFlatNumChildren() {
    int[] ret = {columnOptions.size()};

    for (ParquetWriterOptions opt: columnOptions) {
      int[] b = opt.getFlatNumChildren();
      int[] tmp = new int[ret.length + b.length];
      System.arraycopy(ret, 0, tmp, 0, ret.length);
      System.arraycopy(b, 0, tmp, ret.length, b.length);
      ret = tmp;
    }
    return ret;
  }

  String[] getFlatColumnNames() {
    String[] ret = {columName};
    for (ParquetWriterOptions opt: columnOptions) {
      String[] b = opt.getFlatColumnNames();
      String[] tmp = new String[ret.length + b.length];
      System.arraycopy(ret, 0, tmp, 0, ret.length);
      System.arraycopy(b, 0, tmp, ret.length, b.length);
      ret = tmp;
    }
    return ret;
  }
}
