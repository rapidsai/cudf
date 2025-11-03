/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
  * Options for rolling windows.
 */
public class WindowOptions implements AutoCloseable {

  enum FrameType {ROWS, RANGE}

  /**
   * Extent of (range) window bounds.
   * Analogous to cudf::range_window_bounds::extent_type.
   */
  enum RangeExtentType {
    CURRENT_ROW(0), // Bounds defined as the first/last row that matches the current row.
    BOUNDED(1),     // Bounds defined as the first/last row that falls within
                    // a specified range from the current row.
    UNBOUNDED(2);   // Bounds stretching to the first/last row in the entire group.

    public final int nominalValue;

    RangeExtentType(int n) {
      this.nominalValue = n;
    }
  }

  private final int minPeriods;
  private final Scalar precedingScalar;
  private final Scalar followingScalar;
  private final ColumnVector precedingCol;
  private final ColumnVector followingCol;
  private final int orderByColumnIndex;
  private final boolean orderByOrderAscending;
  private final FrameType frameType;
  private final RangeExtentType precedingBoundsExtent;
  private final RangeExtentType followingBoundsExtent;

  private WindowOptions(Builder builder) {
    this.minPeriods = builder.minPeriods;
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
    this.precedingBoundsExtent = builder.precedingBoundsExtent;
    this.followingBoundsExtent = builder.followingBoundsExtent;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    } else if (other instanceof WindowOptions) {
      WindowOptions o = (WindowOptions) other;
      boolean ret = this.minPeriods == o.minPeriods &&
              this.orderByColumnIndex == o.orderByColumnIndex &&
              this.orderByOrderAscending == o.orderByOrderAscending &&
              this.frameType == o.frameType &&
              this.precedingBoundsExtent == o.precedingBoundsExtent &&
              this.followingBoundsExtent == o.followingBoundsExtent;
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
    ret = 31 * ret + precedingBoundsExtent.hashCode();
    ret = 31 * ret + followingBoundsExtent.hashCode();
    return ret;
  }

  public static Builder builder(){
    return new Builder();
  }

  int getMinPeriods() { return  this.minPeriods; }

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

  boolean isUnboundedPreceding() { return this.precedingBoundsExtent == RangeExtentType.UNBOUNDED; }

  boolean isUnboundedFollowing() { return this.followingBoundsExtent == RangeExtentType.UNBOUNDED; }

  boolean isCurrentRowPreceding() { return this.precedingBoundsExtent == RangeExtentType.CURRENT_ROW; }

  boolean isCurrentRowFollowing() { return this.followingBoundsExtent == RangeExtentType.CURRENT_ROW; }

  RangeExtentType getPrecedingBoundsExtent() { return this.precedingBoundsExtent; }
  RangeExtentType getFollowingBoundsExtent() { return this.followingBoundsExtent; }

  FrameType getFrameType() { return frameType; }

  public static class Builder {
    private int minPeriods = 1;
    // for range window
    private Scalar precedingScalar = null;
    private Scalar followingScalar = null;
    private ColumnVector precedingCol = null;
    private ColumnVector followingCol = null;
    private int orderByColumnIndex = -1;
    private boolean orderByOrderAscending = true;
    private RangeExtentType precedingBoundsExtent = RangeExtentType.BOUNDED;
    private RangeExtentType followingBoundsExtent = RangeExtentType.BOUNDED;

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
     * columns passed in so you have to be sure that the lifetime of the column outlives
     * this operation.
     * @param precedingCol the number of rows preceding the current row and
     *                     precedingCol will be live outside of WindowOptions.
     * @param followingCol the number of rows following the current row and
     *                     following will be live outside of WindowOptions.
     */
    public Builder window(ColumnVector precedingCol, ColumnVector followingCol) {
      if (precedingCol == null || precedingCol.hasNulls()) {
        throw new IllegalArgumentException("preceding cannot be null or have nulls");
      }
      if (followingCol == null || followingCol.hasNulls()) {
        throw new IllegalArgumentException("following cannot be null or have nulls");
      }
      if (precedingBoundsExtent != RangeExtentType.BOUNDED || precedingScalar != null) {
        throw new IllegalStateException("preceding has already been set a different way");
      }
      if (followingBoundsExtent != RangeExtentType.BOUNDED || followingScalar != null) {
        throw new IllegalStateException("following has already been set a different way");
      }
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
      return preceding(precedingScalar).following(followingScalar);
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

    public Builder currentRowPreceding() {
      if (precedingCol != null || precedingScalar != null) {
        throw new IllegalStateException("preceding has already been set a different way");
      }
      this.precedingBoundsExtent = RangeExtentType.CURRENT_ROW;
      return this;
    }

    public Builder currentRowFollowing() {
      if (followingCol != null || followingScalar != null) {
        throw new IllegalStateException("following has already been set a different way");
      }
      this.followingBoundsExtent = RangeExtentType.CURRENT_ROW;
      return this;
    }

    public Builder unboundedPreceding() {
      if (precedingCol != null || precedingScalar != null) {
        throw new IllegalStateException("preceding has already been set a different way");
      }
      this.precedingBoundsExtent = RangeExtentType.UNBOUNDED;
      return this;
    }

    public Builder unboundedFollowing() {
      if (followingCol != null || followingScalar != null) {
        throw new IllegalStateException("following has already been set a different way");
      }
      this.followingBoundsExtent = RangeExtentType.UNBOUNDED;
      return this;
    }

    /**
     * Set the relative number preceding the current row for range window
     * @return this for chaining
     */
    public Builder preceding(Scalar preceding) {
      if (preceding == null || !preceding.isValid()) {
        throw new IllegalArgumentException("preceding cannot be null");
      }
      if (precedingBoundsExtent != RangeExtentType.BOUNDED || precedingCol != null) {
        throw new IllegalStateException("preceding has already been set a different way");
      }
      this.precedingScalar = preceding;
      return this;
    }

    /**
     * Set the relative number following the current row for range window
     * @return this for chaining
     */
    public Builder following(Scalar following) {
      if (following == null || !following.isValid()) {
        throw new IllegalArgumentException("following cannot be null");
      }
      if (followingBoundsExtent != RangeExtentType.BOUNDED || followingCol != null) {
        throw new IllegalStateException("following has already been set a different way");
      }
      this.followingScalar = following;
      return this;
    }

    public WindowOptions build() {
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
