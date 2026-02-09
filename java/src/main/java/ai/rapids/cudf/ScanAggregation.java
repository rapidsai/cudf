/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * An aggregation that can be used for a scan.
 */
public final class ScanAggregation {
  private final Aggregation wrapped;

  private ScanAggregation(Aggregation wrapped) {
    this.wrapped = wrapped;
  }

  long createNativeInstance() {
    return wrapped.createNativeInstance();
  }

  long getDefaultOutput() {
    return wrapped.getDefaultOutput();
  }

  Aggregation getWrapped() {
    return wrapped;
  }

  @Override
  public int hashCode() {
    return wrapped.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    } else if (other instanceof ScanAggregation) {
      ScanAggregation o = (ScanAggregation) other;
      return wrapped.equals(o.wrapped);
    }
    return false;
  }

  /**
   * Sum Aggregation
   */
  public static ScanAggregation sum() {
    return new ScanAggregation(Aggregation.sum());
  }

  /**
   * Product Aggregation.
   */
  public static ScanAggregation product() {
    return new ScanAggregation(Aggregation.product());
  }

  /**
   * Min Aggregation
   */
  public static ScanAggregation min() {
    return new ScanAggregation(Aggregation.min());
  }

  /**
   * Max Aggregation
   */
  public static ScanAggregation max() {
    return new ScanAggregation(Aggregation.max());
  }

  /**
   * Get the row's ranking.
   */
  public static ScanAggregation rank() {
    return new ScanAggregation(Aggregation.rank());
  }

  /**
   * Get the row's dense ranking.
   */
  public static ScanAggregation denseRank() {
    return new ScanAggregation(Aggregation.denseRank());
  }

  /**
   * Get the row's percent rank.
   */
  public static ScanAggregation percentRank() {
    return new ScanAggregation(Aggregation.percentRank());
  }
}
