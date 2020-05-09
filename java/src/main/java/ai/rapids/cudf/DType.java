/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import java.util.EnumSet;

public enum DType {
  EMPTY(0, 0, "NOT SUPPORTED"),
  INT8(1, 1, "byte"),
  INT16(2, 2, "short"),
  INT32(4, 3, "int"),
  INT64(8, 4, "long"),
  FLOAT32(4, 5, "float"),
  FLOAT64(8, 6, "double"),
  /**
   * Byte wise true non-0/false 0.  In general true will be 1.
   */
  BOOL8(1, 7, "bool"),
  /**
   * Days since the UNIX epoch
   */
  TIMESTAMP_DAYS(4, 8, "date32"),
  /**
   * s since the UNIX epoch
   */
  TIMESTAMP_SECONDS(8, 9, "timestamp[s]"),
  /**
   * ms since the UNIX epoch
   */
  TIMESTAMP_MILLISECONDS(8, 10, "timestamp[ms]"),
  /**
   * microseconds since the UNIX epoch
   */
  TIMESTAMP_MICROSECONDS(8, 11, "timestamp[us]"),
  /**
   * ns since the UNIX epoch
   */
  TIMESTAMP_NANOSECONDS(8, 12, "timestamp[ns]"),
  //DICTIONARY32(4, 13, "NO IDEA"),
  STRING(0, 14, "str");

  private static final DType[] D_TYPES = DType.values();
  final int sizeInBytes;
  final int nativeId;
  final String simpleName;

  DType(int sizeInBytes, int nativeId, String simpleName) {
    this.sizeInBytes = sizeInBytes;
    this.nativeId = nativeId;
    this.simpleName = simpleName;
  }

  public boolean isTimestamp() {
    return TIMESTAMPS.contains(this);
  }

  /**
   * Returns true for timestamps with time level resolution, as opposed to day level resolution
   */
  public boolean hasTimeResolution() {
    return TIME_RESOLUTION.contains(this);
  }

  public int getNativeId() {
    return nativeId;
  }

  /**
   * This only works for fixed width types. Variable width types like strings the value is
   * undefined and should be ignored.
   * @return
   */
  public int getSizeInBytes() {
    return sizeInBytes;
  }

  public static DType fromNative(int nativeId) {
    for (DType type : D_TYPES) {
      if (type.nativeId == nativeId) {
        return type;
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a DType");
  }

  private static final EnumSet<DType> TIMESTAMPS = EnumSet.of(
      DType.TIMESTAMP_DAYS,
      DType.TIMESTAMP_SECONDS,
      DType.TIMESTAMP_MILLISECONDS,
      DType.TIMESTAMP_MICROSECONDS,
      DType.TIMESTAMP_NANOSECONDS);

  private static final EnumSet<DType> TIME_RESOLUTION = EnumSet.of(
      DType.TIMESTAMP_SECONDS,
      DType.TIMESTAMP_MILLISECONDS,
      DType.TIMESTAMP_MICROSECONDS,
      DType.TIMESTAMP_NANOSECONDS);
}
