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
