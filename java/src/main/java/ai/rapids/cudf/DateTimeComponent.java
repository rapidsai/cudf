/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Types of datetime components that may be extracted.
 */
public enum DateTimeComponent {
  /**
   * year as an INT16
   */
  YEAR(0),
  /**
   * month 1 - jan, as an INT16
   */
  MONTH(1),
  /**
   * Day of the month as an INT16
   */
  DAY(2),
  /**
   * day of the week, Monday=1, ..., Sunday=7 as an INT16
   */
  WEEKDAY(3),
  /**
   * hour of the day 24-hour clock as an INT16
   */
  HOUR(4),
  /**
   * minutes past the hour as an INT16
   */
  MINUTE(5),
  /**
   * seconds past the minute as an INT16
   */
  SECOND(6),
  /**
   * milliseconds past the seconds as an INT16
   */
  MILLISECOND(7),
  /**
   * microseconds past the millisecond as an INT16
   */
  MICROSECOND(8),
  /**
   * nanoseconds past the microsecond as an INT16
   */
  NANOSECOND(9);

  final int id;
  DateTimeComponent(int id) {
    this.id = id;
  }

  public int getNativeId() {
    return id;
  }
}
