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
  private final boolean[] readBinaryAsString;

  protected ColumnFilterOptions(Builder<?> builder) {
    includeColumnNames = builder.includeColumnNames.toArray(
        new String[builder.includeColumnNames.size()]);
    readBinaryAsString = new boolean[builder.binaryAsStringColumns.size()];
    for (int i = 0 ; i < builder.binaryAsStringColumns.size() ; i++) {
      readBinaryAsString[i] = builder.binaryAsStringColumns.get(i);
    }
  }

  String[] getIncludeColumnNames() {
    return includeColumnNames;
  }

  boolean[] getConvertToBinaryRead() {
    return readBinaryAsString;
  }

  public static class Builder<T extends Builder> {
    final List<String> includeColumnNames = new ArrayList<>();
    final List<Boolean> binaryAsStringColumns = new ArrayList<>();

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    public T includeColumn(String... names) {
      for (String name : names) {
        includeColumnNames.add(name);
        binaryAsStringColumns.add(true);
      }
      return (T) this;
    }

    /**
     * Include this column.
     * @param name the name of the column
     * @param isBinary whether this column is to be read in as binary
     */
    public T includeColumn(String name, boolean isBinary) {
      includeColumnNames.add(name);
      binaryAsStringColumns.add(!isBinary);
      return (T) this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    public T includeColumn(Collection<String> names) {
      for (String name: names) {
        includeColumnNames.add(name);
        binaryAsStringColumns.add(true);
      }
      return (T) this;
    }
  }
}
