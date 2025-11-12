/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

public enum DateTimeRoundingFrequency {
  DAY(0),
  HOUR(1),
  MINUTE(2),
  SECOND(3),
  MILLISECOND(4),
  MICROSECOND(5),
  NANOSECOND(6);

  final int id;
  DateTimeRoundingFrequency(int id) {
    this.id = id;
  }

  public int getNativeId() {
    return id;
  }
}
