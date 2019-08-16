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

  private ORCOptions(Builder builder) {
    super(builder);
    useNumPyTypes = builder.useNumPyTypes;
  }

  boolean usingNumPyTypes() {
    return useNumPyTypes;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private boolean useNumPyTypes = true;

    /**
     * Specify whether the parser should implicitly promote DATE32 and
     * TIMESTAMP columns to DATE64 for compatibility with NumPy.
     *
     * @param useNumPyTypes true to request this conversion, false to avoid.
     * @return builder for chaining
     */
    public Builder withNumPyTypes(boolean useNumPyTypes) {
      this.useNumPyTypes = useNumPyTypes;
      return this;
    }

    public ORCOptions build() { return new ORCOptions(this); }
  }
}
