/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Options for groupby (see cudf::groupby::groupby's constructor)
 */
public class GroupByOptions {

  public static GroupByOptions DEFAULT = new GroupByOptions(new Builder());

  private final boolean ignoreNullKeys;
  private final boolean keysSorted;
  private final boolean[] keysDescending;
  private final boolean[] keysNullSmallest;

  private GroupByOptions(Builder builder) {
    ignoreNullKeys = builder.ignoreNullKeys;
    keysSorted = builder.keysSorted;
    keysDescending = builder.keysDescending;
    keysNullSmallest = builder.keysNullSmallest;
  }

  boolean getIgnoreNullKeys() {
    return ignoreNullKeys;
  }

  boolean getKeySorted() {
    return keysSorted;
  }

  boolean[] getKeysDescending() {
    return keysDescending;
  }

  boolean[] getKeysNullSmallest() {
    return keysNullSmallest;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private boolean ignoreNullKeys = false;
    private boolean keysSorted = false;
    private boolean[] keysDescending = new boolean[0];
    private boolean[] keysNullSmallest = new boolean[0];

    /**
     * If true, the cudf groupby will ignore grouping keys that are null.
     * The default value is false, so a null in the grouping column will produce a
     * group.
     */
    public Builder withIgnoreNullKeys(boolean ignoreNullKeys) {
      this.ignoreNullKeys = ignoreNullKeys;
      return this;
    }

    /**
     * Indicates whether rows in `keys` are already sorted.
     * The default value is false.
     *
     * If the `keys` are already sorted, better performance may be achieved by
     * passing `keysSorted == true` and indicating the ascending/descending
     * order of each column and null order by calling `withKeysDescending` and
     * `withKeysNullSmallest`, respectively.
     */
    public Builder withKeysSorted(boolean keysSorted) {
      this.keysSorted = keysSorted;
      return this;
    }

    /**
     * If `keysSorted == true`, indicates whether each
     * column is ascending/descending. If empty or null, assumes all columns are
     * ascending. Ignored if `keysSorted == false`.
     */
    public Builder withKeysDescending(boolean... keysDescending) {
      if (keysDescending == null) {
        // Use empty array instead of null
        this.keysDescending = new boolean[0];
      } else {
        this.keysDescending = keysDescending;
      }
      return this;
    }

    /**
     * If `keysSorted == true`, indicates the ordering
     * of null values in each column. If empty or null, assumes all columns
     * use 'null smallest'. Ignored if `keysSorted == false`.
     */
    public Builder withKeysNullSmallest(boolean... keysNullSmallest) {
      if (keysNullSmallest == null) {
        // Use empty array instead of null
        this.keysNullSmallest = new boolean[0];
      } else {
        this.keysNullSmallest = keysNullSmallest;
      }
      return this;
    }

    public GroupByOptions build() {
      return new GroupByOptions(this);
    }
  }
}
