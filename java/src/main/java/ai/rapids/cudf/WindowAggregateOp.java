/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
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
 * Operator for window-based aggregation (for analytical functions).
 * It encapsulates the aggregation operation, and window-parameters.
 *
 * Note: An aggregation operation (e.g. SUM) specified with different
 * window parameters (e.g. preceding-window = 5, vs preceding-window = 10)
 * will be deemed as *distinct* window operations.
 */
public class WindowAggregateOp implements Comparable<WindowAggregateOp> {

  private AggregateOp   aggregateOp;
  private WindowOptions windowOptions;

  public WindowAggregateOp(AggregateOp aggregateOp, WindowOptions windowOptions) {
    this.aggregateOp = aggregateOp;
    this.windowOptions = windowOptions;
    assertIsValid();
  }

  public AggregateOp getAggregateOp() {
    return aggregateOp;
  }

  public WindowOptions getWindowOptions() {
    return windowOptions;
  }

  /**
   * Class invariant.
   * @throws IllegalArgumentException if WindowAggregateOp is incomplete
   * @throws UnsupportedOperationException if windowOptions specifies a ColumnVector for precedingCol/followingCol
   */
  private void assertIsValid() {
    if (aggregateOp == null) {
      throw new IllegalArgumentException("Aggregation-operation cannot be null!");
    }

    if (windowOptions == null) {
      throw new IllegalArgumentException("WindowOptions cannot be null!");
    }

    if (windowOptions.getPrecedingCol() != null || windowOptions.getFollowingCol() != null) {
      throw new UnsupportedOperationException("Dynamic windows (via columns) are currently unsupported!");
    }
  }

  @Override
  public int compareTo(WindowAggregateOp rhs) {
    int compareAggOps = this.aggregateOp.compareTo(rhs.aggregateOp);
    if (compareAggOps != 0) {
      return compareAggOps;
    }

    int comparePreceding = Integer.compare(this.windowOptions.getPreceding(), rhs.windowOptions.getPreceding());
    if (comparePreceding != 0) {
      return comparePreceding;
    }

    int compareFollowing = Integer.compare(this.windowOptions.getFollowing(), rhs.windowOptions.getFollowing());
    if (compareFollowing != 0) {
      return compareFollowing;
    }

    int compareTimestampColumnIndex = Integer.compare(this.windowOptions.getTimestampColumnIndex(),
        rhs.windowOptions.getTimestampColumnIndex());

    return compareTimestampColumnIndex != 0? compareTimestampColumnIndex :
        Integer.compare(this.windowOptions.getMinPeriods(), rhs.windowOptions.getMinPeriods());
  }
}
