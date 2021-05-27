/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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
 * Options for reading a parquet file
 */
public class ParquetOptions extends ColumnFilterOptions {

  public static ParquetOptions DEFAULT = new ParquetOptions(new Builder());

  private final DType unit;

  private final boolean strictDecimalType;


  private ParquetOptions(Builder builder) {
    super(builder);
    unit = builder.unit;
    strictDecimalType = builder.strictDecimalType;
  }

  DType timeUnit() {
    return unit;
  }

  boolean isStrictDecimalType() {
    return strictDecimalType;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private DType unit = DType.EMPTY;
    private boolean strictDecimalType = false;

    /**
     * Specify the time unit to use when returning timestamps.
     * @param unit default unit of time specified by the user
     * @return builder for chaining
     */
    public Builder withTimeUnit(DType unit) {
      assert unit.isTimestampType();
      this.unit = unit;
      return this;
    }

    /**
     * Specify how to deal with decimal columns who are not backed by INT32/64 while reading.
     * @param strictDecimalType whether strictly reading all decimal columns as fixed-point decimal type
     * @return builder for chaining
     */
    public Builder enableStrictDecimalType(boolean strictDecimalType) {
      this.strictDecimalType = strictDecimalType;
      return this;
    }

    public ParquetOptions build() {
      return new ParquetOptions(this);
    }
  }
}
