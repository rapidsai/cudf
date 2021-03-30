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

import java.util.ArrayList;
import java.util.List;

/**
 * Per column settings for writing Parquet files.
 */
public class ParquetColumnWriterOptions implements RapidsSerializable {

  public static class Builder {
    private String columnName;
    private boolean isTimestampTypeInt96 = false;
    private int precision;
    private boolean isNullable = false;
    private List<ParquetColumnWriterOptions> childColumnOptions = new ArrayList<>();

    /**
     * Name of this column
     * @param name
     * @return this for chaining.
     */
    public Builder withColumnName(String name) {
      this.columnName = name;
      return this;
    }

    /**
     * Whether this column can have null values
     * @param isNullable
     * @return this for chaining.
     */
    public Builder isNullable(boolean isNullable) {
      this.isNullable = isNullable;
      return this;
    }

    /**
     * Set whether the timestamps should be written in INT96
     * @return this for chaining.
     */
    public Builder withTimestampInt96(boolean int96) {
      this.isTimestampTypeInt96 = int96;
      return this;
    }

    /**
     * @param precision a value for this column if decimal
     * @return this for chaining.
     */
    public Builder withDecimalPrecision(int precision) {
      this.precision = precision;
      return this;
    }

    /**
     * Create a column with these options
     * @return this for chaining
     */
    public Builder withColumnOptions(ParquetColumnWriterOptions columnOptions) {
      childColumnOptions.add(columnOptions);
      return this;
    }

    public ParquetColumnWriterOptions build() {
      return new ParquetColumnWriterOptions(this);
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  private ParquetColumnWriterOptions(Builder builder) {
    this.isTimestampTypeInt96 = builder.isTimestampTypeInt96;
    this.precision = builder.precision;
    this.isNullable = builder.isNullable;
    this.columName = builder.columnName;
    this.childColumnOptions = builder.childColumnOptions
        .toArray(new ParquetColumnWriterOptions[builder.childColumnOptions.size()]);
  }
  /**
   * Return if the column can have null values
   */
  public String getColumName() {
    return columName;
  }

  /**
   * Return if the column can have null values
   */
  public boolean isNullable() {
    return isNullable;
  }

  /**
   * Return the precision for this column
   */
  public int getPrecision() {
    return precision;
  }

  @Override
  public List<Boolean> getFlatIsTimeTypeInt96() {
    List<Boolean> a = new ArrayList<>();
    a.add(isTimestampTypeInt96);
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatIsTimeTypeInt96());
    }
    return a;
  }

  @Override
  public List<Integer> getFlatPrecision() {
    List<Integer> a = new ArrayList<>();
    a.add(getPrecision());
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatPrecision());
    }
    return a;
  }

  @Override
  public List<Boolean> getFlatIsNullable() {
    List<Boolean> a = new ArrayList<>();
    a.add(isNullable());
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatIsNullable());
    }
    return a;
  }

  @Override
  public List<String> getFlatColumnNames() {
    List<String> a = new ArrayList<>();
    a.add(getColumName());
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatColumnNames());
    }
    return a;
  }

  @Override
  public List<Integer> getFlatNumChildren() {
    List<Integer> a = new ArrayList<>();
    a.add(childColumnOptions.length);
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatNumChildren());
    }
    return a;
  }

  /**
   * Returns true if the writer is expected to write timestamps in INT96
   */
  public boolean isTimestampTypeInt96() {
    return isTimestampTypeInt96;
  }

  public ParquetColumnWriterOptions[] getChildColumnOptions() {
    return childColumnOptions;
  }

  private boolean isTimestampTypeInt96;

  private int precision;

  private boolean isNullable;

  private String columName;

  private ParquetColumnWriterOptions[] childColumnOptions;
}
