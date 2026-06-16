/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */
package ai.rapids.cudf;

enum MaskState {
  UNALLOCATED(0),
  UNINITIALIZED(1),
  ALL_VALID(2),
  ALL_NULL(3);

  private static final MaskState[] MASK_STATES = MaskState.values();
  final int nativeId;

  MaskState(int nativeId) {
    this.nativeId = nativeId;
  }

  static MaskState fromNative(int nativeId) {
    for (MaskState type : MASK_STATES) {
      if (type.nativeId == nativeId) {
        return type;
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a MaskState");
  }
}
