/*
 *
 *  Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
public class WindowOptions implements AutoCloseable {

  enum FrameType {ROWS, RANGE}

  private final int minPeriods;
  private final int preceding;
  private final int following;
  private final Scalar precedingScalar;
  private final Scalar followingScalar;
  private final ColumnVector precedingCol;
  private final ColumnVector followingCol;
  private final int orderByColumnIndex;
  private final boolean orderByOrderAscending;
  private final FrameType frameType;
  private final boolean isUnboundedPreceding;
  private final boolean isUnboundedFollowing;


  private WindowOptions(Builder builder) {
    this.minPeriods = builder.minPeriods;
    this.preceding = builder.preceding;
    this.following = builder.following;
    this.precedingScalar = builder.precedingScalar;
    if (precedingScalar != null) {
      precedingScalar.incRefCount();
    }
    this.followingScalar = builder.followingScalar;
    if (followingScalar != null) {
      followingScalar.incRefCount();
    }
    this.precedingCol = builder.precedingCol;
    if (precedingCol != null) {
      precedingCol.incRefCount();
    }
    this.followingCol = builder.followingCol;
    if (followingCol != null) {
      followingCol.incRefCount();
    }
    this.orderByColumnIndex = builder.orderByColumnIndex;
    this.orderByOrderAscending = builder.orderByOrderAscending;
    this.frameType = orderByColumnIndex == -1? FrameType.ROWS : FrameType.RANGE;
    this.isUnboundedPreceding = builder.isUnboundedPreceding;
    this.isUnboundedFollowing = builder.isUnboundedFollowing;

  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    } else if (other instanceof WindowOptions) {
      WindowOptions o = (WindowOptions) other;
      boolean ret = this.preceding == o.preceding &&
              this.following == o.following &&
              this.minPeriods == o.minPeriods &&
              this.orderByColumnIndex == o.orderByColumnIndex &&
              this.orderByOrderAscending == o.orderByOrderAscending &&
              this.frameType == o.frameType &&
              this.isUnboundedPreceding == o.isUnboundedPreceding &&
              this.isUnboundedFollowing == o.isUnboundedFollowing;
      if (precedingCol != null) {
        ret = ret && precedingCol.equals(o.precedingCol);
      }
      if (followingCol != null) {
        ret = ret && followingCol.equals(o.followingCol);
      }
      if (precedingScalar != null) {
        ret = ret && precedingScalar.equals(o.precedingScalar);
      }
      if (followingScalar != null) {
        ret = ret && followingScalar.equals(o.followingScalar);
      }
      return ret;
    }
    return false;
  }

  @Override
  public int hashCode() {
    int ret = 7;
    ret = 31 * ret + preceding;
    ret = 31 * ret + following;
    ret = 31 * ret + minPeriods;
    ret = 31 * ret + orderByColumnIndex;
    ret = 31 * ret + Boolean.hashCode(orderByOrderAscending);
    ret = 31 * ret + frameType.hashCode();
    if (precedingCol != null) {
      ret = 31 * ret + precedingCol.hashCode();
    }
    if (followingCol != null) {
      ret = 31 * ret + followingCol.hashCode();
    }
    if (precedingScalar != null) {
      ret = 31 * ret + precedingScalar.hashCode();
    }
    if (followingScalar != null) {
      ret = 31 * ret + followingScalar.hashCode();
    }
    ret = 31 * ret + Boolean.hashCode(isUnboundedPreceding);
    ret = 31 * ret + Boolean.hashCode(isUnboundedFollowing);
    return ret;
  }

  public static Builder builder(){
    return new Builder();
  }

  int getMinPeriods() { return  this.minPeriods; }

  int getPreceding() { return this.preceding; }

  int getFollowing() { return this.following; }

  Scalar getPrecedingScalar() { return this.precedingScalar; }

  Scalar getFollowingScalar() { return this.followingScalar; }

  ColumnVector getPrecedingCol() { return precedingCol; }

  ColumnVector getFollowingCol() { return this.followingCol; }

  @Deprecated
  int getTimestampColumnIndex() { return getOrderByColumnIndex(); }

  int getOrderByColumnIndex() { return this.orderByColumnIndex; }

  @Deprecated
  boolean isTimestampOrderAscending() { return isOrderByOrderAscending(); }

  boolean isOrderByOrderAscending() { return this.orderByOrderAscending; }

  boolean isUnboundedPreceding() { return this.isUnboundedPreceding; }

  boolean isUnboundedFollowing() { return this.isUnboundedFollowing; }

  FrameType getFrameType() { return frameType; }

  public static class Builder {
    private int minPeriods = 1;
    private int preceding = 0;
    private int following = 1;
    // for range window
    private Scalar precedingScalar = null;
    private Scalar followingScalar = null;
    boolean staticSet = false;
    private ColumnVector precedingCol = null;
    private ColumnVector followingCol = null;
    private int orderByColumnIndex = -1;
    private boolean orderByOrderAscending = true;
    private boolean isUnboundedPreceding = false;
    private boolean isUnboundedFollowing = false;

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
     * @param precedingCol the number of rows preceding the current row and
     *                     precedingCol will be live outside of WindowOptions.
     * @param followingCol the number of rows following the current row and
     *                     following will be live outside of WindowOptions.
     */
    public Builder window(ColumnVector precedingCol, ColumnVector followingCol) {
      assert (precedingCol != null && precedingCol.getNullCount() == 0);
      assert (followingCol != null && followingCol.getNullCount() == 0);
      this.precedingCol = precedingCol;
      this.followingCol = followingCol;
      return this;
    }

    /**
     * Set the size of the range window.
     * @param precedingScalar the relative number preceding the current row and
     *                        the precedingScalar will be live outside of WindowOptions.
     * @param followingScalar the relative number following the current row and
     *                        the followingScalar will be live outside of WindowOptions
     */
    public Builder window(Scalar precedingScalar, Scalar followingScalar) {
      assert (precedingScalar != null && precedingScalar.isValid());
      assert (followingScalar != null && followingScalar.isValid());
      this.precedingScalar = precedingScalar;
      this.followingScalar = followingScalar;
      return this;
    }

    /**
     * @deprecated Use orderByColumnIndex(int index)
     */
    @Deprecated
    public Builder timestampColumnIndex(int index) {
      return orderByColumnIndex(index);
    }

    public Builder orderByColumnIndex(int index) {
      this.orderByColumnIndex = index;
      return this;
    }

    /**
     * @deprecated Use orderByAscending()
     */
    @Deprecated
    public Builder timestampAscending() {
      return orderByAscending();
    }

    public Builder orderByAscending() {
      this.orderByOrderAscending = true;
      return this;
    }

    public Builder orderByDescending() {
      this.orderByOrderAscending = false;
      return this;
    }

    /**
     * @deprecated Use orderByDescending()
     */
    @Deprecated
    public Builder timestampDescending() {
      return orderByDescending();
    }

    public Builder unboundedPreceding() {
      this.isUnboundedPreceding = true;
      return this;
    }

    public Builder unboundedFollowing() {
      this.isUnboundedFollowing = true;
      return this;
    }

    public Builder preceding(int preceding) {
      this.preceding = preceding;
      return this;
    }

    public Builder following(int following) {
      this.following = following;
      return this;
    }

    /**
     * Set the relative number preceding the current row for range window
     * @param preceding
     * @return Builder
     */
    public Builder preceding(Scalar preceding) {
      this.precedingScalar = preceding;
      return this;
    }

    /**
     * Set the relative number following the current row for range window
     * @param following
     * @return Builder
     */
    public Builder following(Scalar following) {
      this.followingScalar = following;
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

  public synchronized WindowOptions incRefCount() {
    if (precedingScalar != null) {
      precedingScalar.incRefCount();
    }
    if (followingScalar != null) {
      followingScalar.incRefCount();
    }
    if (precedingCol != null) {
      precedingCol.incRefCount();
    }
    if (followingCol != null) {
      followingCol.incRefCount();
    }
    return this;
  }

  @Override
  public void close() {
    if (precedingScalar != null) {
      precedingScalar.close();
    }
    if (followingScalar != null) {
      followingScalar.close();
    }
    if (precedingCol != null) {
      precedingCol.close();
    }
    if (followingCol != null) {
      followingCol.close();
    }
  }
}