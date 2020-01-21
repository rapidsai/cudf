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
 * Options for group by (see cudf::groupby::Options)
 */
public class GroupByOptions {

  public static GroupByOptions DEFAULT = new GroupByOptions(new Builder());

  private final boolean ignoreNullKeys;

  private GroupByOptions(Builder builder) {
    ignoreNullKeys = builder.ignoreNullKeys;
  }

  boolean getIgnoreNullKeys() {
    return ignoreNullKeys;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private boolean ignoreNullKeys = false;

    /**
     * If true, the cudf groupby will ignore grouping keys that are null.
     * The default value is false, so a null in the grouping column will produce a
     * group.
     */
    public Builder withIgnoreNullKeys(boolean ignoreNullKeys) {
      this.ignoreNullKeys = ignoreNullKeys;
      return this;
    }

    public GroupByOptions build() {
      return new GroupByOptions(this);
    }
  }
}
