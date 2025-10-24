/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.io.Serializable;

/**
 * Provides the ordering for specific columns.
 */
public final class OrderByArg implements Serializable {
  final int index;
  final boolean isDescending;
  final boolean isNullSmallest;

  OrderByArg(int index, boolean isDescending, boolean isNullSmallest) {
    this.index = index;
    this.isDescending = isDescending;
    this.isNullSmallest = isNullSmallest;
  }

  public static OrderByArg asc(final int index) {
    return new OrderByArg(index, false, false);
  }

  public static OrderByArg desc(final int index) {
    return new OrderByArg(index, true, false);
  }

  public static OrderByArg asc(final int index, final boolean isNullSmallest) {
    return new OrderByArg(index, false, isNullSmallest);
  }

  public static OrderByArg desc(final int index, final boolean isNullSmallest) {
    return new OrderByArg(index, true, isNullSmallest);
  }

  @Override
  public String toString() {
    return "ORDER BY " + index +
        (isDescending ? " DESC " : " ASC ") +
        (isNullSmallest ? "NULL SMALLEST" : "NULL LARGEST");
  }
}
