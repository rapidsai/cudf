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
 * Arguments for Window function
 * The window size, the number of observations and window size in forward direction
 * can be static or dynamic (varying for each element).
 * windowSize    - The static rolling window size. If windowCol = NULL,
 *                   output_col[i] accumulates values from input_col[i-window+1]
 *                   to input_col[i] inclusive
 * minPeriods    - Minimum number of observations in window required to
 *                   have a value, otherwise 0 is stored in the valid bit mask for
 *                   element i. If minPeriodsCol != NULL, then minimum number of
 *                   observations for element i is obtained from minPeriodsCol[i]
 * forwardWindow - The static window size in the forward direction. If
 *                   forwardWindowCol = NULL, output_col[i] accumulates values
 *                   from input_col[i] to input_col[i+forward_window] inclusive
 * aggType       - The rolling window aggregtion type (sum, max, min, etc.)
 * windowCol     - The window size values, windowCol[i] specifies window
 *                   size for element i. If windowCol = NULL, then windowSize is
 *                   used as the static window size for all elements
 * minPeriodsCol - The minimum number of observation values, minPeriodsCol[i]
 *                   specifies minimum number of observations for element i.
 *                   If minPeriodsCol = NULL, then minPeriods is used as
 *                   the static value for all elements
 *forwardWindowCol-The forward window size values, forwardWindowCol[i] specifies
 *                   forward window size for element i. If forwardWindowCol = NULL,
 *                   then forwardWindow is used as the static forward window size
 *                   for all elements
 * WindowOptions does not take ownership of ColumnVectors passed in,
 * so the caller is responsible for closing the vectors after the call using
 * WindowOptions is complete.
 * Currently there is a bug where if minPeriods is not satisfied for a particular row,
 * then that row is not updated with null.
 * Being tracked in https://github.com/rapidsai/cudf/issues/2689
 */
public class WindowOptions {

  public static WindowOptions DEFAULT = new WindowOptions(new Builder());

  private final int windowSize;
  private final int minPeriods;
  private final int forwardWindow;
  private final AggregateOp aggType;
  private final ColumnVector windowCol;
  private final ColumnVector minPeriodsCol;
  private final ColumnVector forwardWindowCol;


  private WindowOptions(Builder builder) {
    this.windowSize = builder.windowSize;
    this.minPeriods = builder.minPeriods;
    this.forwardWindow = builder.forwardWindow;
    this.aggType = builder.aggType;
    this.windowCol = builder.windowCol;
    this.minPeriodsCol = builder.minPeriodsCol;
    this.forwardWindowCol = builder.forwardWindowCol;
  }

  public static Builder builder(){
    return new Builder();
  }

  int getWindow() { return this.windowSize; }

  int getMinPeriods() { return  this.minPeriods; }

  int getForwardWindow() { return this.forwardWindow; }

  AggregateOp getAggType() { return this.aggType; }

  ColumnVector getWindowCol() { return  windowCol; }

  ColumnVector getMinPeriodsCol() { return  this.minPeriodsCol; }

  ColumnVector getForwardWindowCol() { return this.forwardWindowCol; }

  public static class Builder {
    private int windowSize = -1;
    private int minPeriods = -1;
    private int forwardWindow = -1;
    private AggregateOp aggType = AggregateOp.SUM;
    private ColumnVector windowCol = null;
    private ColumnVector minPeriodsCol = null;
    private ColumnVector forwardWindowCol = null;

    /**
     * Set the static rolling window size.
     */
    public Builder windowSize(int windowSize) {
      if (windowSize < 0 ) {
        throw  new IllegalArgumentException("Window size must be non negative");
      }
      this.windowSize = windowSize;
      return this;
    }

    /**
     * Set the static minimum number of observation required to evaluate element.
     */
    public Builder minPeriods(int minPeriods) {
      if (minPeriods < 0 ) {
        throw  new IllegalArgumentException("Minimum observations must be non negative");
      }
      this.minPeriods = minPeriods;
      return this;
    }

    /**
     * Set the static window size in forward direction.
     */
    public Builder forwardWindow(int forwardWindow) {
      if (forwardWindow < 0 ) {
        throw  new IllegalArgumentException("Forward window size must be non negative");
      }
      this.forwardWindow = forwardWindow;
      return this;
    }

    /**
     * Set the rolling window aggregation type.
     */
    public Builder aggType(AggregateOp aggType) {
      if (aggType.nativeId < 0 || aggType.nativeId > 4) {
        throw new IllegalArgumentException("Invalid Aggregation Type");
      }
      this.aggType = aggType;
      return this;
    }

    /**
     * Set the window size values for each element in the column.
     * The caller owns the vector which is passed in below and is responsible for
     * it's lifecycle.
     */
    public Builder windowCol(ColumnVector windowCol) {
      assert (windowCol == null || windowCol.getNullCount() == 0);
      this.windowCol = windowCol;
      return this;
    }

    /**
     * Set the minimum number of observations for each element in the column.
     * The caller owns the vector which is passed in below and is responsible for
     * it's lifecycle.
     */
    public Builder minPeriodsCol(ColumnVector minPeriodsCol) {
      assert (minPeriodsCol == null || minPeriodsCol.getNullCount() == 0);
      this.minPeriodsCol = minPeriodsCol;
      return this;
    }

    /**
     * Set the forward window size values for each element in the column.
     * The caller owns the vector which is passed in below and is responsible for
     * it's lifecycle.
     */
    public Builder forwardWindowCol(ColumnVector forwardWindowCol) {
      assert (forwardWindowCol == null || forwardWindowCol.getNullCount() == 0);
      this.forwardWindowCol = forwardWindowCol;
      return this;
    }

    public WindowOptions build() {
      if (windowCol != null && windowSize != -1) {
        throw new IllegalArgumentException("Either windowSize or windCol should be provided");
      }
      if (minPeriodsCol != null && minPeriods != -1) {
        throw new IllegalArgumentException("Either minPeriods or minPeriodsCol should be provided");
      }
      if (forwardWindowCol != null && forwardWindow != -1) {
        throw new IllegalArgumentException
          ("Either forwardWindow or forwardWindowCol should be provided");
      }
      return new WindowOptions(this);
    }
  }
}