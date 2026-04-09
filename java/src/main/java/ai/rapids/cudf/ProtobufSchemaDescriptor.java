/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.HashSet;
import java.util.Set;

/**
 * Immutable descriptor for a flattened protobuf schema, grouping the parallel arrays
 * that describe field structure, types, defaults, and enum metadata.
 *
 * <p>All arrays provided to the constructor are defensively copied to guarantee immutability.
 */
public final class ProtobufSchemaDescriptor implements java.io.Serializable {
  private static final long serialVersionUID = 1L;
  private static final int MAX_FIELD_NUMBER = (1 << 29) - 1;
  private static final int MAX_NESTING_DEPTH = 10;
  private static final int STRUCT_TYPE_ID = DType.STRUCT.getTypeId().getNativeId();
  private static final int STRING_TYPE_ID = DType.STRING.getTypeId().getNativeId();
  private static final int LIST_TYPE_ID = DType.LIST.getTypeId().getNativeId();
  private static final int BOOL8_TYPE_ID = DType.BOOL8.getTypeId().getNativeId();
  private static final int INT32_TYPE_ID = DType.INT32.getTypeId().getNativeId();
  private static final int UINT32_TYPE_ID = DType.UINT32.getTypeId().getNativeId();
  private static final int INT64_TYPE_ID = DType.INT64.getTypeId().getNativeId();
  private static final int UINT64_TYPE_ID = DType.UINT64.getTypeId().getNativeId();
  private static final int FLOAT32_TYPE_ID = DType.FLOAT32.getTypeId().getNativeId();
  private static final int FLOAT64_TYPE_ID = DType.FLOAT64.getTypeId().getNativeId();

  // Encoding constants
  public static final int ENC_DEFAULT = 0;
  public static final int ENC_FIXED = 1;
  public static final int ENC_ZIGZAG = 2;
  public static final int ENC_ENUM_STRING = 3;

  // Wire type constants
  public static final int WT_VARINT = 0;
  public static final int WT_64BIT = 1;
  public static final int WT_LEN = 2;
  public static final int WT_32BIT = 5;

  final int[] fieldNumbers;
  final int[] parentIndices;
  final int[] depthLevels;
  final int[] wireTypes;
  final int[] outputTypeIds;
  final int[] encodings;
  final boolean[] isRepeated;
  final boolean[] isRequired;
  final boolean[] hasDefaultValue;
  final long[] defaultInts;
  final double[] defaultFloats;
  final boolean[] defaultBools;
  final byte[][] defaultStrings;
  final int[][] enumValidValues;
  final byte[][][] enumNames;

  /**
   * @throws IllegalArgumentException if any array is null, arrays have mismatched lengths,
   *         field numbers are out of range, or encoding values are invalid.
   */
  public ProtobufSchemaDescriptor(
      int[] fieldNumbers,
      int[] parentIndices,
      int[] depthLevels,
      int[] wireTypes,
      int[] outputTypeIds,
      int[] encodings,
      boolean[] isRepeated,
      boolean[] isRequired,
      boolean[] hasDefaultValue,
      long[] defaultInts,
      double[] defaultFloats,
      boolean[] defaultBools,
      byte[][] defaultStrings,
      int[][] enumValidValues,
      byte[][][] enumNames) {

    validate(fieldNumbers, parentIndices, depthLevels, wireTypes, outputTypeIds,
        encodings, isRepeated, isRequired, hasDefaultValue, defaultInts,
        defaultFloats, defaultBools, defaultStrings, enumValidValues, enumNames);

    this.fieldNumbers = fieldNumbers.clone();
    this.parentIndices = parentIndices.clone();
    this.depthLevels = depthLevels.clone();
    this.wireTypes = wireTypes.clone();
    this.outputTypeIds = outputTypeIds.clone();
    this.encodings = encodings.clone();
    this.isRepeated = isRepeated.clone();
    this.isRequired = isRequired.clone();
    this.hasDefaultValue = hasDefaultValue.clone();
    this.defaultInts = defaultInts.clone();
    this.defaultFloats = defaultFloats.clone();
    this.defaultBools = defaultBools.clone();
    this.defaultStrings = deepCopy(defaultStrings);
    this.enumValidValues = deepCopy(enumValidValues);
    this.enumNames = deepCopy(enumNames);
  }

  public int numFields() { return fieldNumbers.length; }

  private void readObject(java.io.ObjectInputStream in)
      throws java.io.IOException, ClassNotFoundException {
    in.defaultReadObject();
    try {
      validate(fieldNumbers, parentIndices, depthLevels, wireTypes, outputTypeIds,
          encodings, isRepeated, isRequired, hasDefaultValue, defaultInts,
          defaultFloats, defaultBools, defaultStrings, enumValidValues, enumNames);
    } catch (IllegalArgumentException e) {
      java.io.InvalidObjectException ioe = new java.io.InvalidObjectException(e.getMessage());
      ioe.initCause(e);
      throw ioe;
    }
  }

  private static byte[][] deepCopy(byte[][] src) {
    byte[][] dst = new byte[src.length][];
    for (int i = 0; i < src.length; i++) {
      dst[i] = src[i] != null ? src[i].clone() : null;
    }
    return dst;
  }

  private static int[][] deepCopy(int[][] src) {
    int[][] dst = new int[src.length][];
    for (int i = 0; i < src.length; i++) {
      dst[i] = src[i] != null ? src[i].clone() : null;
    }
    return dst;
  }

  private static byte[][][] deepCopy(byte[][][] src) {
    byte[][][] dst = new byte[src.length][][];
    for (int i = 0; i < src.length; i++) {
      if (src[i] == null) continue;
      dst[i] = new byte[src[i].length][];
      for (int j = 0; j < src[i].length; j++) {
        dst[i][j] = src[i][j] != null ? src[i][j].clone() : null;
      }
    }
    return dst;
  }

  private static void validate(
      int[] fieldNumbers, int[] parentIndices, int[] depthLevels,
      int[] wireTypes, int[] outputTypeIds, int[] encodings,
      boolean[] isRepeated, boolean[] isRequired, boolean[] hasDefaultValue,
      long[] defaultInts, double[] defaultFloats, boolean[] defaultBools,
      byte[][] defaultStrings, int[][] enumValidValues, byte[][][] enumNames) {

    if (fieldNumbers == null || parentIndices == null || depthLevels == null ||
        wireTypes == null || outputTypeIds == null || encodings == null ||
        isRepeated == null || isRequired == null || hasDefaultValue == null ||
        defaultInts == null || defaultFloats == null || defaultBools == null ||
        defaultStrings == null || enumValidValues == null || enumNames == null) {
      throw new IllegalArgumentException("All schema arrays must be non-null");
    }

    int n = fieldNumbers.length;
    if (parentIndices.length != n || depthLevels.length != n ||
        wireTypes.length != n || outputTypeIds.length != n ||
        encodings.length != n || isRepeated.length != n ||
        isRequired.length != n || hasDefaultValue.length != n ||
        defaultInts.length != n || defaultFloats.length != n ||
        defaultBools.length != n || defaultStrings.length != n ||
        enumValidValues.length != n || enumNames.length != n) {
      throw new IllegalArgumentException("All schema arrays must have the same length");
    }

    Set<Long> seenFieldNumbers = new HashSet<>();
    for (int i = 0; i < n; i++) {
      if (fieldNumbers[i] <= 0 || fieldNumbers[i] > MAX_FIELD_NUMBER) {
        throw new IllegalArgumentException(
            "Invalid field number at index " + i + ": " + fieldNumbers[i]);
      }
      if (depthLevels[i] < 0 || depthLevels[i] >= MAX_NESTING_DEPTH) {
        throw new IllegalArgumentException(
            "Invalid depth at index " + i + ": " + depthLevels[i]);
      }
      int pi = parentIndices[i];
      if (pi < -1 || pi >= i) {
        throw new IllegalArgumentException(
            "Invalid parent index at index " + i + ": " + pi);
      }
      if (pi == -1) {
        if (depthLevels[i] != 0) {
          throw new IllegalArgumentException(
              "Top-level field at index " + i + " must have depth 0");
        }
      } else {
        if (outputTypeIds[pi] != STRUCT_TYPE_ID) {
          throw new IllegalArgumentException(
              "Parent at index " + pi + " for field " + i + " must be STRUCT");
        }
      }
      long fieldKey = (((long) pi) << 32) | (fieldNumbers[i] & 0xFFFFFFFFL);
      if (!seenFieldNumbers.add(fieldKey)) {
        throw new IllegalArgumentException(
            "Duplicate field number " + fieldNumbers[i] + " under parent " + pi);
      }
      int wt = wireTypes[i];
      if (wt != WT_VARINT && wt != WT_64BIT && wt != WT_LEN && wt != WT_32BIT) {
        throw new IllegalArgumentException("Invalid wire type at index " + i + ": " + wt);
      }
      int enc = encodings[i];
      if (enc < ENC_DEFAULT || enc > ENC_ENUM_STRING) {
        throw new IllegalArgumentException("Invalid encoding at index " + i + ": " + enc);
      }
    }
  }
}
