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

import java.util.ArrayList;
import java.util.List;

/**
 * The schema of data to be read in.
 */
public class Schema {
  public static final Schema INFERRED = new Schema();
  private final List<String> names;
  private final List<String> typeNames;
  private final List<DType> types;

  private Schema(List<String> names, List<String> typeNames, List<DType> types) {
    this.names = new ArrayList<>(names);
    this.typeNames = new ArrayList<>(typeNames);
    this.types = types;
  }

  /**
   * Inferred schema.
   */
  private Schema() {
    names = null;
    typeNames = null;
    types = null;
  }

  public static Builder builder() {
    return new Builder();
  }

  public String[] getColumnNames() {
    if (names == null) {
      return null;
    }
    return names.toArray(new String[names.size()]);
  }

  String[] getTypesAsStrings() {
    if (typeNames == null) {
      return null;
    }
    return typeNames.toArray(new String[typeNames.size()]);
  }

  /**
   * Guess at the size of an output table based off of the number of expected rows.
   * If there are any strings in the data a default size of 10 bytes per string is used.
   * @param numRows the number of rows to use in the estimate.
   * @return the estimated number of bytes needed to read in this schema.
   * @throws IllegalStateException if no type information is available, an INFERRED schema
   */
  public long guessTableSize(int numRows) {
    return guessTableSize(numRows, 10);
  }

  /**
   * Guess at the size of an output table based off of the number of expected rows.
   * @param numRows the number of rows to use in the estimate.
   * @param avgStringSize the estimated average size of a string.
   * @return the estimated number of bytes needed to read in this schema.
   */
  public long guessTableSize(int numRows, int avgStringSize) {
    if (types == null) {
      throw new IllegalStateException("No type information is available to guess the output size");
    }
    long total = 0;
    for (DType type: types) {
      if (type == DType.STRING_CATEGORY || type == DType.STRING) {
        total += avgStringSize * numRows;
        total += 4 * (numRows + 1); // Offsets
      } else {
        total += type.sizeInBytes * numRows;
      }
      // Assume that there is validity
      total += (numRows + 7) / 8;
    }
    return total;
  }

  public static class Builder {
    private List<String> names = new ArrayList<>();
    private List<String> typeNames = new ArrayList<>();
    private List<DType> types = new ArrayList<>();

    public Builder column(DType type, String name) {
      types.add(type);
      typeNames.add(type.simpleName);
      names.add(name);
      return this;
    }

    public Schema build() {
      return new Schema(names, typeNames, types);
    }
  }
}
