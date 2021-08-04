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
