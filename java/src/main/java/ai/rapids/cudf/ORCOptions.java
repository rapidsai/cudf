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
import java.util.Arrays;
import java.util.List;

/**
 * Options for reading a ORC file
 */
public class ORCOptions extends ColumnFilterOptions {

  public static ORCOptions DEFAULT = new ORCOptions(new Builder());

  private final boolean useNumPyTypes;
  private final DType unit;
  private final String[] decimal128Columns;

  private ORCOptions(Builder builder) {
    super(builder);
    decimal128Columns = builder.decimal128Columns.toArray(new String[0]);
    useNumPyTypes = builder.useNumPyTypes;
    unit = builder.unit;
  }

  boolean usingNumPyTypes() {
    return useNumPyTypes;
  }

  DType timeUnit() {
    return unit;
  }

  String[] getDecimal128Columns() {
    return decimal128Columns;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private boolean useNumPyTypes = true;
    private DType unit = DType.EMPTY;

    final List<String> decimal128Columns = new ArrayList<>();

    /**
     * Specify whether the parser should implicitly promote TIMESTAMP_DAYS
     * columns to TIMESTAMP_MILLISECONDS for compatibility with NumPy.
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
     * @param unit default unit of time specified by the user
     * @return builder for chaining
     */
    public ORCOptions.Builder withTimeUnit(DType unit) {
      assert unit.isTimestampType();
      this.unit = unit;
      return this;
    }

    /**
     * Specify decimal columns which shall be read as DECIMAL128. Otherwise, decimal columns
     * will be read as DECIMAL64 by default in ORC.
     *
     * In terms of child columns of nested types, their parents need to be prepended as prefix
     * of the column name, in case of ambiguity. For struct columns, the names of child columns
     * are formatted as `{struct_col_name}.{child_col_name}`. For list columns, the data(child)
     * columns are named as `{list_col_name}.1`.
     *
     * @param names names of columns which read as DECIMAL128
     * @return builder for chaining
     */
    public Builder decimal128Column(String... names) {
      decimal128Columns.addAll(Arrays.asList(names));
      return this;
    }

    public ORCOptions build() { return new ORCOptions(this); }
  }
}
