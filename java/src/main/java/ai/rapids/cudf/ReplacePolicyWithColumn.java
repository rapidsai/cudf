/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * A replacement policy for a specific column
 */
public class ReplacePolicyWithColumn {
  final int column;
  final ReplacePolicy policy;

  ReplacePolicyWithColumn(int column, ReplacePolicy policy) {
    this.column = column;
    this.policy = policy;
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof ReplacePolicyWithColumn)) {
      return false;
    }
    ReplacePolicyWithColumn ro = (ReplacePolicyWithColumn)other;
    return this.column == ro.column && this.policy.equals(ro.policy);
  }

  @Override
  public int hashCode() {
    return 31 * column + policy.hashCode();
  }
}
