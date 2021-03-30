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

import java.util.Arrays;
import java.util.List;

/**
 * Per column settings for writing Parquet files.
 */
public class ParquetTimestampColumnWriterOptions implements RapidsSerializable {

  public static class Builder {
    private String columnName;
    private boolean isTimestampTypeInt96 = false;
    private boolean isNullable = false;

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

    public ParquetTimestampColumnWriterOptions build() {
      return new ParquetTimestampColumnWriterOptions(this);
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  private ParquetTimestampColumnWriterOptions(Builder builder) {
    this.isTimestampTypeInt96 = builder.isTimestampTypeInt96;
    this.isNullable = builder.isNullable;
    this.columName = builder.columnName;
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

  @Override
  public List<Boolean> getFlatIsTimeTypeInt96() {
    return Arrays.asList(isTimestampTypeInt96);
  }

  @Override
  public List<Integer> getFlatPrecision() {
    return Arrays.asList(0);
  }

  @Override
  public List<Boolean> getFlatIsNullable() {
    return Arrays.asList(isNullable);
  }

  @Override
  public List<String> getFlatColumnNames() {
    return Arrays.asList(getColumName());
  }

  @Override
  public List<Integer> getFlatNumChildren() {
    return Arrays.asList(0);
  }

  /**
   * Returns true if the writer is expected to write timestamps in INT96
   */
  public boolean isTimestampTypeInt96() {
    return isTimestampTypeInt96;
  }

  private boolean isTimestampTypeInt96;

  private boolean isNullable;

  private String columName;
}
