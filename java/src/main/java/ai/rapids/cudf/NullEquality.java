/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * How should nulls be compared in an operation.
 */
public enum NullEquality {
  UNEQUAL(false),
  EQUAL(true);

  NullEquality(boolean nullsEqual) {
    this.nullsEqual = nullsEqual;
  }

  final boolean nullsEqual;
}
