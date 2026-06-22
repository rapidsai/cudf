/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  ULONGLONG(7),
  BITS(0xff);

  private static final CompressionType[] types = CompressionType.values();

  final int nativeId;

  CompressionType(int nativeId) {
    this.nativeId = nativeId;
  }

  /** Lookup the CompressionType that corresponds to the specified native identifier */
  public static CompressionType fromNativeId(int id) {
    for (CompressionType type : types) {
      if (type.nativeId == id) {
        return type;
      }
    }
    throw new IllegalArgumentException("Unknown compression type ID: " + id);
  }

  /** Get the native code identifier for the type */
  public final int toNativeId() {
    return nativeId;
  }
}
