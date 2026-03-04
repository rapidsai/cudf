/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Hash algorithm identifiers, mirroring native enum cudf::hash_id
 */
public enum HashType {
  IDENTITY(0),
  MURMUR3(1);

  private static final HashType[] HASH_TYPES = HashType.values();
  final int nativeId;

  HashType(int nativeId) {
    this.nativeId = nativeId;
  }

  public int getNativeId() {
    return nativeId;
  }

  public static HashType fromNative(int nativeId) {
    for (HashType type : HASH_TYPES) {
      if (type.nativeId == nativeId) {
        return type;
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a HashType");
  }
}
