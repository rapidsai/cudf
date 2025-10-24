/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Policy to specify the position of replacement values relative to null rows.
 */
public enum ReplacePolicy {
  /**
   * The replacement value is the first non-null value preceding the null row.
   */
  PRECEDING(true),
  /**
   * The replacement value is the first non-null value following the null row.
   */
  FOLLOWING(false);

  ReplacePolicy(boolean isPreceding) {
    this.isPreceding = isPreceding;
  }

  final boolean isPreceding;

  /**
   * Indicate which column the replacement should happen on.
   */
  public ReplacePolicyWithColumn onColumn(int columnNumber) {
    return new ReplacePolicyWithColumn(columnNumber, this);
  }
}
