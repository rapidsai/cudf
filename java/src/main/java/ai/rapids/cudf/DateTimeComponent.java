/*
 *
 *  Copyright (c) 2024, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
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
