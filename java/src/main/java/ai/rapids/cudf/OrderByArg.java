/*
 *
 *  Copyright (c) 2021, NVIDIA CORPORATION.
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
