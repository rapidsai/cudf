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
 * Options for reading a ORC file
 */
public class ORCOptions extends ColumnFilterOptions {

  public static ORCOptions DEFAULT = new ORCOptions(new Builder());

  private final boolean useNumPyTypes;
  private final TimeUnit unit;
  private final int rowGuess;

  private ORCOptions(Builder builder) {
    super(builder);
    useNumPyTypes = builder.useNumPyTypes;
    unit = builder.unit;
    rowGuess = builder.rowGuess;
  }

  boolean usingNumPyTypes() {
    return useNumPyTypes;
  }

  TimeUnit timeUnit() {
    return unit;
  }

  int getRowGuess() {
    return rowGuess;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private boolean useNumPyTypes = true;
    private TimeUnit unit = TimeUnit.NONE;
    private int rowGuess = -1;

    /**
     * A guess of how many rows will be loaded. This is totally optional and only used for
     * estimating the memory usage of loading the data.
     * @param guess the number of rows to be loaded as a guess.
     * @return this for chaining.
     */
    public Builder withRowGuess(int guess) {
      this.rowGuess = guess;
      return this;
    }

    /**
     * Specify whether the parser should implicitly promote DATE32
     * column to DATE64 for compatibility with NumPy.
     *
     * @param useNumPyTypes true to request this conversion, false to avoid.
     * @return builder for chaining
     */
    public Builder withNumPyTypes(boolean useNumPyTypes) {
      this.useNumPyTypes = useNumPyTypes;
      return this;
    }

    /**
     * Specify the time unit to use when returning timestamps.
     * @param unit TimeUnit specified by the user
     * @return builder for chaining
     */
    public Builder withTimeUnit(TimeUnit unit) {
      this.unit = unit;
      return this;
    }

    public ORCOptions build() { return new ORCOptions(this); }
  }
}
