/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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
 * Interpolation method to use when the desired quantile lies between
 * two data points i and j.
 */
public enum QuantileMethod {

  /**
   * Linear interpolation between i and j
   */
  LINEAR(0),
  /**
   * Lower data point (i)
   */
  LOWER(1),
  /**
   * Higher data point (j)
   */
  HIGHER(2),
  /**
   * (i + j)/2
   */
  MIDPOINT(3),
  /**
   * i or j, whichever is nearest
   */
  NEAREST(4);

  final int nativeId;

  QuantileMethod(int nativeId) {
    this.nativeId = nativeId;
  }
}
