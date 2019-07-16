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
 * Method used to calculate the quantiles
 */
public enum QuantileMethod {

  LINEAR(0),
  LOWER(1),
  HIGHER(2),
  MIDPOINT(3),
  NEAREST(4);

  final int nativeId;

  QuantileMethod(int nativeId) {
    this.nativeId = nativeId;
  }
}
