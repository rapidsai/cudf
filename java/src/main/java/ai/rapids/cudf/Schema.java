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
import java.util.LinkedList;
import java.util.List;

/**
 * The schema of data to be read in.
 */
public class Schema {
  public static final Schema INFERRED = new Schema();
  private final List<String> names;
  private final List<String> types;

  private Schema(List<String> names, List<String> types) {
    this.names = new ArrayList<>(names);
    this.types = new ArrayList<>(types);
  }

  /**
   * Inferred schema.
   */
  private Schema() {
    names = null;
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
    if (types == null) {
      return null;
    }
    return types.toArray(new String[types.size()]);
  }

  public static class Builder {
    private List<String> names = new LinkedList<>();
    private List<String> types = new LinkedList<>();

    public Builder column(DType type, String name) {
      types.add(type.simpleName);
      names.add(name);
      return this;
    }

    public Schema build() {
      return new Schema(names, types);
    }
  }
}
