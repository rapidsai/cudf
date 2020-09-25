/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

package ai.rapids.cudf.nvcomp;

/** Enumeration of data types that can be compressed. */
public enum CompressionType {
  CHAR(0),
  UCHAR(1),
  SHORT(2),
  USHORT(3),
  INT(4),
  UINT(5),
  LONGLONG(6),
  ULONGLONG(7);

  final int nativeId;

  CompressionType(int nativeId) {
    this.nativeId = nativeId;
  }

  public final int toNativeId() {
    return nativeId;
  }
}
