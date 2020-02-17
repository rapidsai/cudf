/*
 *
 *  Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
 * Aggregate operations on a column
 */
enum AggregateOp {
  SUM(0),
  MIN(1),
  MAX(2),
  COUNT(3),
  MEAN(4),
  MEDIAN(5),
  // TODO Quantile
  ARGMAX(7),
  ARGMIN(8),
  PRODUCT(9),
  SUMOFSQUARES(10),
  VAR(11), // This can take a delta degrees of freedom
  STD(12), // This can take a delta degrees of freedom
  ANY(13),
  ALL(14);

  final int nativeId;

  AggregateOp(int nativeId) {this.nativeId = nativeId;}
}
