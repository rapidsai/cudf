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
 * A replacement policy for a specific column
 */
public class ReplacePolicyWithColumn {
  final int column;
  final ReplacePolicy policy;

  ReplacePolicyWithColumn(int column, ReplacePolicy policy) {
    this.column = column;
    this.policy = policy;
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof ReplacePolicyWithColumn)) {
      return false;
    }
    ReplacePolicyWithColumn ro = (ReplacePolicyWithColumn)other;
    return this.column == ro.column && this.policy.equals(ro.policy);
  }

  @Override
  public int hashCode() {
    return 31 * column + policy.hashCode();
  }
}
