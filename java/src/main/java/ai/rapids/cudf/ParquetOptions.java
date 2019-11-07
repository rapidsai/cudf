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

  private final TimeUnit unit;

  private ParquetOptions(Builder builder) {
    super(builder);
    unit = builder.unit;
  }

  TimeUnit timeUnit() {
    return unit;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private TimeUnit unit = TimeUnit.NONE;

    /**
     * Specify the time unit to use when returning timestamps.
     * @param unit TimeUnit specified by the user
     * @return builder for chaining
     */
    public Builder withTimeUnit(TimeUnit unit) {
      this.unit = unit;
      return this;
    }

    public ParquetOptions build() {
      return new ParquetOptions(this);
    }
  }
}