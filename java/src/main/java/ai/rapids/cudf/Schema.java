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

  long getRowsSizeGuess() {
    long total = 0;
    if (types == null) {
      total += 1000; // This is really just a hack, who knows
    } else {
      for (DType type: types) {
        if (type == DType.STRING_CATEGORY || type == DType.STRING) {
          total += 4 + 10; // 10 chars and a single offset
        } else {
          total += type.sizeInBytes;
        }
      }
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
