/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.cudf;

/**
 * Reduction operations on a column
 */
public enum ReductionOp {
  SUM(0),
  MIN(1),
  MAX(2),
  PRODUCT(3),
  SUMOFSQUARES(4);

  final int nativeId;

  ReductionOp(int nativeId) {
    this.nativeId = nativeId;
  }
}
