/*
 *
 *  Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

  private final DType topLevelType;
  private final List<String> childNames;
  private final List<Schema> childSchemas;
  private boolean flattened = false;
  private String[] flattenedNames;
  private DType[] flattenedTypes;
  private int[] flattenedCounts;

  private Schema(DType topLevelType,
                 List<String> childNames,
                 List<Schema> childSchemas) {
    this.topLevelType = topLevelType;
    this.childNames = childNames;
    this.childSchemas = childSchemas;
  }

  /**
   * Inferred schema.
   */
  private Schema() {
    topLevelType = null;
    childNames = null;
    childSchemas = null;
  }

  /**
   * Get the schema of a child element. Note that an inferred schema will have no children.
   * @param i the index of the child to read.
   * @return the new Schema
   * @throws IndexOutOfBoundsException if the index is not in the range of children.
   */
  public Schema getChild(int i) {
    if (childSchemas == null) {
      throw new IndexOutOfBoundsException("There are 0 children in this schema");
    }
    return childSchemas.get(i);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append(topLevelType);
    if (topLevelType == DType.STRUCT) {
      sb.append("{");
      if (childNames != null) {
        for (int i = 0; i < childNames.size(); i++) {
          if (i != 0) {
            sb.append(", ");
          }
          sb.append(childNames.get(i));
          sb.append(": ");
          sb.append(childSchemas.get(i));
        }
      }
      sb.append("}");
    } else if (topLevelType == DType.LIST) {
      sb.append("[");
      if (childNames != null) {
        for (int i = 0; i < childNames.size(); i++) {
          if (i != 0) {
            sb.append(", ");
          }
          sb.append(childSchemas.get(i));
        }
      }
      sb.append("]");
    }
    return sb.toString();
  }

  private void flattenIfNeeded() {
    if (!flattened) {
      int flatLen = flattenedLength(0);
      if (flatLen == 0) {
        flattenedNames = null;
        flattenedTypes = null;
        flattenedCounts = null;
      } else {
        String[] names = new String[flatLen];
        DType[] types = new DType[flatLen];
        int[] counts = new int[flatLen];
        collectFlattened(names, types, counts, 0);
        flattenedNames = names;
        flattenedTypes = types;
        flattenedCounts = counts;
      }
      flattened = true;
    }
  }

  private int flattenedLength(int startingLength) {
    if (childSchemas != null) {
      for (Schema child: childSchemas) {
        startingLength++;
        startingLength = child.flattenedLength(startingLength);
      }
    }
    return startingLength;
  }

  private int collectFlattened(String[] names, DType[] types, int[] counts, int offset) {
    if (childSchemas != null) {
      for (int i = 0; i < childSchemas.size(); i++) {
        Schema child = childSchemas.get(i);
        names[offset] = childNames.get(i);
        types[offset] = child.topLevelType;
        if (child.childNames != null) {
          counts[offset] = child.childNames.size();
        } else {
          counts[offset] = 0;
        }
        offset++;
        offset = this.childSchemas.get(i).collectFlattened(names, types, counts, offset);
      }
    }
    return offset;
  }

  public static Builder builder() {
    return new Builder(DType.STRUCT);
  }

  public String[] getFlattenedColumnNames() {
    flattenIfNeeded();
    return flattenedNames;
  }

  public String[] getColumnNames() {
    if (childNames == null) {
      return null;
    }
    return childNames.toArray(new String[childNames.size()]);
  }

  public boolean isNested() {
    return childSchemas != null && childSchemas.size() > 0;
  }

  /**
   * This is really for a top level struct schema where it is nested, but
   * for things like CSV we care that it does not have any children that are also
   * nested.
   */
  public boolean hasNestedChildren() {
    if (childSchemas != null) {
      for (Schema child: childSchemas) {
        if (child.isNested()) {
          return true;
        }
      }
    }
    return false;
  }

  int[] getFlattenedTypeIds() {
    flattenIfNeeded();
    if (flattenedTypes == null) {
      return null;
    }
    int[] ret = new int[flattenedTypes.length];
    for (int i = 0; i < flattenedTypes.length; i++) {
      ret[i] = flattenedTypes[i].getTypeId().nativeId;
    }
    return ret;
  }

  int[] getFlattenedTypeScales() {
    flattenIfNeeded();
    if (flattenedTypes == null) {
      return null;
    }
    int[] ret = new int[flattenedTypes.length];
    for (int i = 0; i < flattenedTypes.length; i++) {
      ret[i] = flattenedTypes[i].getScale();
    }
    return ret;
  }

  DType[] getFlattenedTypes() {
    flattenIfNeeded();
    return flattenedTypes;
  }

  public DType[] getChildTypes() {
    if (childSchemas == null) {
      return null;
    }
    DType[] ret = new DType[childSchemas.size()];
    for (int i = 0; i < ret.length; i++) {
      ret[i] = childSchemas.get(i).topLevelType;
    }
    return ret;
  }

  int[] getFlattenedNumChildren() {
    flattenIfNeeded();
    return flattenedCounts;
  }

  public DType getType() {
    return topLevelType;
  }

  /**
   * Check to see if the schema includes a struct at all.
   * @return true if this or any one of its descendants contains a struct, else false.
   */
  public boolean isStructOrHasStructDescendant() {
    if (DType.STRUCT == topLevelType) {
      return true;
    } else if (DType.LIST == topLevelType) {
      return childSchemas.stream().anyMatch(Schema::isStructOrHasStructDescendant);
    }
    return false;
  }

  public static class Builder {
    private final DType topLevelType;
    private final List<String> names;
    private final List<Builder> types;

    private Builder(DType topLevelType) {
      this.topLevelType = topLevelType;
      if (topLevelType == DType.STRUCT || topLevelType == DType.LIST) {
        // There can be children
        names = new ArrayList<>();
        types = new ArrayList<>();
      } else {
        names = null;
        types = null;
      }
    }

    /**
     * Add a new column
     * @param type the type of column to add
     * @param name the name of the column to add (Ignored for list types)
     * @return the builder for the new column. This should really only be used when the type
     * passed in is a LIST or a STRUCT.
     */
    public Builder addColumn(DType type, String name) {
      if (names == null) {
        throw new IllegalStateException("A column of type " + topLevelType +
            " cannot have children");
      }
      if (topLevelType == DType.LIST && names.size() > 0) {
        throw new IllegalStateException("A LIST column can only have one child");
      }
      if (names.contains(name)) {
        throw new IllegalStateException("Cannot add duplicate names to a schema");
      }
      Builder ret = new Builder(type);
      types.add(ret);
      names.add(name);
      return ret;
    }

    /**
     * Adds a single column to the current schema. addColumn is preferred as it can be used
     * to support nested types.
     * @param type the type of the column.
     * @param name the name of the column.
     * @return this for chaining.
     */
    public Builder column(DType type, String name) {
      addColumn(type, name);
      return this;
    }

    public Schema build() {
      List<Schema> children = null;
      if (types != null) {
        children = new ArrayList<>(types.size());
        for (Builder b: types) {
          children.add(b.build());
        }
      }
      return new Schema(topLevelType, names, children);
    }
  }
}
