/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * How should NaNs be compared in an operation. In floating point there are multiple
 * different binary representations for NaN.
 */
public enum NaNEquality {
  /**
   * No NaN representation is considered equal to any NaN representation, even for the
   * exact same representation.
   */
  UNEQUAL(false),
  /**
   * All representations of NaN are considered to be equal.
   */
  ALL_EQUAL(true);

  NaNEquality(boolean nansEqual) {
    this.nansEqual = nansEqual;
  }

  final boolean nansEqual;
}
