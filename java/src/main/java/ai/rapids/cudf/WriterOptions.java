/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
