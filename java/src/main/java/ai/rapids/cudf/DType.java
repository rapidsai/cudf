/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

public enum DType {
  INVALID(0, 0, "invalid"),
  INT8(1, 1, "int"),
  INT16(2, 2, "short"),
  INT32(4, 3, "int32"),
  INT64(8, 4, "int64"),
  FLOAT32(4, 5, "float32"),
  FLOAT64(8, 6, "float64"),
  /**
   * Byte wise true non-0/false 0.  In general true will be 1.
   */
  BOOL8(1, 7, "bool"),
  /**
   * Days since the UNIX epoch
   */
  DATE32(4, 8, "date32"),
  /**
   * ms since the UNIX epoch
   */
  DATE64(8, 9, "date64"),
  /**
   * Exact timestamp encoded with int64 since the UNIX epoch (Default unit ms)
   */
  TIMESTAMP(8, 10, "timestamp"),
  STRING(0, 12, "str"),
  // IMPLEMENTATION DETAIL: The sizeInBytes is 4 for STRING_CATEGORY because the dictionary is
  // stored in the data pointer as ints.  This makes some of the code in ColumnVector common
  // for STRING_CATEGORY.
  /**
   * Strings, but stored on the device with a dictionary for compression.
   */
  STRING_CATEGORY(4, 13, "not-supported");

  private static final DType[] D_TYPES = DType.values();
  final int sizeInBytes;
  final int nativeId;
  final String simpleName;

  DType(int sizeInBytes, int nativeId, String simpleName) {
    this.sizeInBytes = sizeInBytes;
    this.nativeId = nativeId;
    this.simpleName = simpleName;
  }

  static DType fromNative(int nativeId) {
    for (DType type : D_TYPES) {
      if (type.nativeId == nativeId) {
        return type;
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a DType");
  }
}
