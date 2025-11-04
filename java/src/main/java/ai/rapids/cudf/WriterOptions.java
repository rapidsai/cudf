/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class WriterOptions {

  private final String[] columnNames;
  private final boolean[] columnNullability;

  <T extends WriterBuilder> WriterOptions(T builder) {
    columnNames = (String[]) builder.columnNames.toArray(new String[builder.columnNames.size()]);
    columnNullability = new boolean[builder.columnNullability.size()];
    for (int i = 0; i < builder.columnNullability.size(); i++) {
      columnNullability[i] = (boolean)builder.columnNullability.get(i);
    }
  }

  public String[] getColumnNames() {
    return columnNames;
  }

  public boolean[] getColumnNullability() {
    return columnNullability;
  }

  protected static class WriterBuilder<T extends WriterBuilder> {
    final List<String> columnNames = new ArrayList<>();
    final List<Boolean> columnNullability = new ArrayList<>();

    /**
     * Add column name(s). For Parquet column names are not optional.
     * @param columnNames
     */
    public T withColumnNames(String... columnNames) {
      this.columnNames.addAll(Arrays.asList(columnNames));
      for (int i = 0; i < columnNames.length; i++) {
        this.columnNullability.add(true);
      }
      return (T) this;
    }

    /**
     * Add column name that is not nullable
     * @param columnNames
     */
    public T withNotNullableColumnNames(String... columnNames) {
      this.columnNames.addAll(Arrays.asList(columnNames));
      for (int i = 0; i < columnNames.length; i++) {
        this.columnNullability.add(false);
      }
      return (T) this;
    }
  }
}
