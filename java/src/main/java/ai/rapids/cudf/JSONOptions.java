/*
 *
 *  Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import java.util.Collection;

/**
 * Options for reading in JSON encoded data.
 */
public final class JSONOptions extends ColumnFilterOptions {

  public static JSONOptions DEFAULT = new JSONOptions(builder());

  private final boolean dayFirst;
  private final boolean lines;

  private JSONOptions(Builder builder) {
    super(builder);
    dayFirst = builder.dayFirst;
    lines = builder.lines;
  }

  public boolean isDayFirst() {
    return dayFirst;
  }

  public boolean isLines() {
    return lines;
  }

  @Override
  String[] getIncludeColumnNames() {
    throw new UnsupportedOperationException("JSON reader didn't support column prune");
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder  extends ColumnFilterOptions.Builder<JSONOptions.Builder> {
    private boolean dayFirst = false;
    private boolean lines = true;

    /**
     * Whether to parse dates as DD/MM versus MM/DD
     * @param dayFirst true: DD/MM, false, MM/DD
     * @return
     */
    public Builder withDayFirst(boolean dayFirst) {
      this.dayFirst = dayFirst;
      return this;
    }

    /**
     * Whether to read the file as a json object per line
     * @param perLine true: per line, false: multi-line
     * @return builder for chaining
     */
    public Builder withLines(boolean perLine) {
      assert perLine == true : "Cudf does not support multi-line";
      this.lines = perLine;
      return this;
    }

    @Override
    public Builder includeColumn(String... names) {
      throw new UnsupportedOperationException("JSON reader didn't support column prune");
    }

    @Override
    public Builder includeColumn(Collection<String> names) {
      throw new UnsupportedOperationException("JSON reader didn't support column prune");
    }

    public JSONOptions build() {
      return new JSONOptions(this);
    }
  }
}
