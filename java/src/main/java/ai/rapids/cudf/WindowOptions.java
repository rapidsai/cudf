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
  * Options for rolling windows.
 */
public class WindowOptions {
  private final int preceding;
  private final int minPeriods;
  private final int following;
  private final ColumnVector precedingCol;
  private final ColumnVector followingCol;

  private WindowOptions(Builder builder) {
    this.preceding = builder.preceding;
    this.minPeriods = builder.minPeriods;
    this.following = builder.following;
    this.precedingCol = builder.precedingCol;
    this.followingCol = builder.followingCol;
  }

  public static Builder builder(){
    return new Builder();
  }

  int getMinPeriods() { return  this.minPeriods; }

  int getPreceding() { return this.preceding; }

  int getFollowing() { return this.following; }

  ColumnVector getPrecedingCol() { return precedingCol; }

  ColumnVector getFollowingCol() { return this.followingCol; }

  public static class Builder {
    private int minPeriods = 1;
    private int preceding = 0;
    private int following = 1;
    boolean staticSet = false;
    private ColumnVector precedingCol = null;
    private ColumnVector followingCol = null;

    /**
     * Set the minimum number of observation required to evaluate an element.  If there are not
     * enough elements for a given window a null is placed in the result instead.
     */
    public Builder minPeriods(int minPeriods) {
      if (minPeriods < 0 ) {
        throw  new IllegalArgumentException("Minimum observations must be non negative");
      }
      this.minPeriods = minPeriods;
      return this;
    }

    /**
     * Set the size of the window, one entry per row. This does not take ownership of the
     * columns passed in so you have to be sure that the life time of the column outlives
     * this operation.
     * @param precedingCol the number of rows preceding the current row.
     * @param followingCol the number of rows following the current row.
     */
    public Builder window(ColumnVector precedingCol, ColumnVector followingCol) {
      assert (precedingCol != null && precedingCol.getNullCount() == 0);
      assert (followingCol != null && followingCol.getNullCount() == 0);
      this.precedingCol = precedingCol;
      this.followingCol = followingCol;
      return this;
    }

    /**
     * Set the size of the window.
     * @param preceding the number of rows preceding the current row
     * @param following the number of rows following the current row.
     */
    public Builder window(int preceding, int following) {
      this.preceding = preceding;
      this.following = following;
      staticSet = true;
      return this;
    }

    public WindowOptions build() {
      if (staticSet && precedingCol != null) {
        throw new IllegalArgumentException("Cannot set both a static window and a non-static window");
      }
      return new WindowOptions(this);
    }
  }
}