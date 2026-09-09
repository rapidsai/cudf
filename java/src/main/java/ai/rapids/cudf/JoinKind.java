/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/** Join semantics to apply when filtering join gather maps. */
public enum JoinKind {
  /** Retain only row pairs that satisfy the condition. */
  INNER(0),
  /** Retain every left row, using an invalid right index when no pair satisfies the condition. */
  LEFT(1),
  /** Retain every row from both sides, splitting row pairs that do not satisfy the condition. */
  FULL(2);

  final int nativeId;

  JoinKind(int nativeId) {
    this.nativeId = nativeId;
  }
}
