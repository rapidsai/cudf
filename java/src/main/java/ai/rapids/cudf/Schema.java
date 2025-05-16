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
import java.util.stream.Collectors;

/**
 * The schema of data to be read in.
 */
public class Schema {
  public static final Schema INFERRED = new Schema();

  private final DType topLevelType;

  /**
   * Default value for precision value, when it is not specified or the column type is not decimal.
   */
  private static final int UNKNOWN_PRECISION = -1;

  /**
   * Store precision for the top level column, only applicable if the column is a decimal type.
   * <p/>
   * This variable is not designed to be used by any libcudf's APIs since libcudf does not support
   * precisions for fixed point numbers.
   * Instead, it is used only to pass down the precision values from Spark's DecimalType to the
   * JNI level, where some JNI functions require these values to perform their operations.
   */
  private final int topLevelPrecision;

  private final List<String> childNames;
  private final List<Schema> childSchemas;
  private boolean flattened = false;
  private String[] flattenedNames;
  private DType[] flattenedTypes;
  private int[] flattenedPrecisions;
  private int[] flattenedCounts;

  private Schema(DType topLevelType,
                 int topLevelPrecision,
                 List<String> childNames,
                 List<Schema> childSchemas) {
    this.topLevelType = topLevelType;
    this.topLevelPrecision = topLevelPrecision;
    this.childNames = childNames;
    this.childSchemas = childSchemas;
  }

  private Schema(DType topLevelType,
                 List<String> childNames,
                 List<Schema> childSchemas) {
    this(topLevelType, UNKNOWN_PRECISION, childNames, childSchemas);
  }

  /**
   * Inferred schema.
   */
  private Schema() {
    topLevelType = null;
    topLevelPrecision = UNKNOWN_PRECISION;
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
        flattenedPrecisions = null;
        flattenedCounts = null;
      } else {
        String[] names = new String[flatLen];
        DType[] types = new DType[flatLen];
        int[] precisions = new int[flatLen];
        int[] counts = new int[flatLen];
        collectFlattened(names, types, precisions, counts, 0);
        flattenedNames = names;
        flattenedTypes = types;
        flattenedPrecisions = precisions;
        flattenedCounts = counts;
      }
      flattened = true;
    }
  }

  private int flattenedLength(int startingLength) {
    if (childSchemas != null) {
      for (Schema child : childSchemas) {
        startingLength++;
        startingLength = child.flattenedLength(startingLength);
      }
    }
    return startingLength;
  }

  private int collectFlattened(String[] names, DType[] types, int[] precisions, int[] counts, int offset) {
    if (childSchemas != null) {
      for (int i = 0; i < childSchemas.size(); i++) {
        Schema child = childSchemas.get(i);
        names[offset] = childNames.get(i);
        types[offset] = child.topLevelType;
        precisions[offset] = child.topLevelPrecision;
        if (child.childNames != null) {
          counts[offset] = child.childNames.size();
        } else {
          counts[offset] = 0;
        }
        offset++;
        offset = this.childSchemas.get(i).collectFlattened(names, types, precisions, counts, offset);
      }
    }
    return offset;
  }

  public static Builder builder() {
    return new Builder(DType.STRUCT);
  }

  /**
   * Get names of the columns flattened from all levels in schema by depth-first traversal.
   * @return An array containing names of all columns in schema.
   */
  public String[] getFlattenedColumnNames() {
    flattenIfNeeded();
    return flattenedNames;
  }

  /**
   * Get names of the top level child columns in schema.
   * @return An array containing names of top level child columns.
   */
  public String[] getColumnNames() {
    if (childNames == null) {
      return null;
    }
    return childNames.toArray(new String[childNames.size()]);
  }

  /**
   * Check if the schema is nested (i.e., top level type is LIST or STRUCT).
   * @return true if the schema is nested, false otherwise.
   */
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
      for (Schema child : childSchemas) {
        if (child.isNested()) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Get type ids of the columns flattened from all levels in schema by depth-first traversal.
   * @return An array containing type ids of all columns in schema.
   */
  public int[] getFlattenedTypeIds() {
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

  /**
   * Get scales of the columns' types flattened from all levels in schema by depth-first traversal.
   * @return An array containing type scales of all columns in schema.
   */
  public int[] getFlattenedTypeScales() {
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

  /**
   * Get decimal precisions of the columns' types flattened from all levels in schema by
   * depth-first traversal.
   * <p/>
   * This is used to pass down the decimal precisions from Spark to only the JNI layer, where
   * some JNI functions require precision values to perform their operations.
   * Decimal precisions should not be consumed by any libcudf's APIs since libcudf does not
   * support precisions for fixed point numbers.
   *
   * @return An array containing decimal precision of all columns in schema.
   */
  public int[] getFlattenedDecimalPrecisions() {
    flattenIfNeeded();
    return flattenedPrecisions;
  }

  /**
   * Get the types of the columns in schema flattened from all levels by depth-first traversal.
   * @return An array containing types of all columns in schema.
   */
  public DType[] getFlattenedTypes() {
    flattenIfNeeded();
    return flattenedTypes;
  }

  /**
   * Get types of the top level child columns in schema.
   * @return An array containing types of top level child columns.
   */
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

  /**
   * Get number of top level child columns in schema.
   * @return Number of child columns.
   */
  public int getNumChildren() {
    if (childSchemas == null) {
      return 0;
    }
    return childSchemas.size();
  }

  /**
   * Get numbers of child columns for each level in schema.
   * @return Numbers of child columns for all levels flattened by depth-first traversal.
   */
  public int[] getFlattenedNumChildren() {
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

  public HostColumnVector.DataType asHostDataType() {
    if (topLevelType == DType.LIST) {
      assert (childSchemas != null && childSchemas.size() == 1);
      HostColumnVector.DataType element = childSchemas.get(0).asHostDataType();
      return new HostColumnVector.ListType(true, element);
    } else if (topLevelType == DType.STRUCT) {
      if (childSchemas == null) {
        return new HostColumnVector.StructType(true);
      } else {
        List<HostColumnVector.DataType> childTypes =
            childSchemas.stream().map(Schema::asHostDataType).collect(Collectors.toList());
        return new HostColumnVector.StructType(true, childTypes);
      }
    } else {
      return new HostColumnVector.BasicType(true, topLevelType);
    }
  }

  public static class Builder {
    private final DType topLevelType;
    private final int topLevelPrecision;
    private final List<String> names;
    private final List<Builder> types;

    private Builder(DType topLevelType, int topLevelPrecision) {
      this.topLevelType = topLevelType;
      this.topLevelPrecision = topLevelPrecision;
      if (topLevelType == DType.STRUCT || topLevelType == DType.LIST) {
        // There can be children
        names = new ArrayList<>();
        types = new ArrayList<>();
      } else {
        names = null;
        types = null;
      }
    }

    private Builder(DType topLevelType) {
      this(topLevelType, UNKNOWN_PRECISION);
    }

    /**
     * Add a new column
     * @param type the type of column to add
     * @param name the name of the column to add (Ignored for list types)
     * @param precision the decimal precision, only applicable for decimal types
     * @return the builder for the new column. This should really only be used when the type
     * passed in is a LIST or a STRUCT.
     */
    public Builder addColumn(DType type, String name, int precision) {
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
      Builder ret = new Builder(type, precision);
      types.add(ret);
      names.add(name);
      return ret;
    }

    public Builder addColumn(DType type, String name) {
      return addColumn(type, name, UNKNOWN_PRECISION);
    }

    /**
     * Adds a single column to the current schema. addColumn is preferred as it can be used
     * to support nested types.
     * @param type the type of the column.
     * @param name the name of the column.
     * @param precision the decimal precision, only applicable for decimal types.
     * @return this for chaining.
     */
    public Builder column(DType type, String name, int precision) {
      addColumn(type, name, precision);
      return this;
    }

    public Builder column(DType type, String name) {
      addColumn(type, name, UNKNOWN_PRECISION);
      return this;
    }

    public Schema build() {
      List<Schema> children = null;
      if (types != null) {
        children = new ArrayList<>(types.size());
        for (Builder b : types) {
          children.add(b.build());
        }
      }
      return new Schema(topLevelType, topLevelPrecision, names, children);
    }
  }
}
