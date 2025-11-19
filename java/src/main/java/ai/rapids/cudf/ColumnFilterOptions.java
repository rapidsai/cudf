/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Base options class for input formats that can filter columns.
 */
public abstract class ColumnFilterOptions {
  // Names of the columns to be returned (other columns are skipped)
  // If empty all columns are returned.
  private final String[] includeColumnNames;

  protected ColumnFilterOptions(Builder<?> builder) {
    includeColumnNames = builder.includeColumnNames.toArray(
        new String[builder.includeColumnNames.size()]);
  }

  String[] getIncludeColumnNames() {
    return includeColumnNames;
  }

  public static class Builder<T extends Builder> {
    final List<String> includeColumnNames = new ArrayList<>();

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    public T includeColumn(String... names) {
      for (String name : names) {
        includeColumnNames.add(name);
      }
      return (T) this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    public T includeColumn(Collection<String> names) {
      includeColumnNames.addAll(names);
      return (T) this;
    }
  }
}
