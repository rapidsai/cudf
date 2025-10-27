/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * An aggregation that can be used for a reduce.
 */
public final class SegmentedReductionAggregation {
  private final Aggregation wrapped;

  private SegmentedReductionAggregation(Aggregation wrapped) {
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
    } else if (other instanceof SegmentedReductionAggregation) {
      SegmentedReductionAggregation o = (SegmentedReductionAggregation) other;
      return wrapped.equals(o.wrapped);
    }
    return false;
  }

  /**
   * Sum Aggregation
   */
  public static SegmentedReductionAggregation sum() {
    return new SegmentedReductionAggregation(Aggregation.sum());
  }

  /**
   * Product Aggregation.
   */
  public static SegmentedReductionAggregation product() {
    return new SegmentedReductionAggregation(Aggregation.product());
  }

  /**
   * Min Aggregation
   */
  public static SegmentedReductionAggregation min() {
    return new SegmentedReductionAggregation(Aggregation.min());
  }

  /**
   * Max Aggregation
   */
  public static SegmentedReductionAggregation max() {
    return new SegmentedReductionAggregation(Aggregation.max());
  }

  /**
   * Any reduction. Produces a true or 1, depending on the output type,
   * if any of the elements in the range are true or non-zero, otherwise produces a false or 0.
   * Null values are skipped.
   */
  public static SegmentedReductionAggregation any() {
    return new SegmentedReductionAggregation(Aggregation.any());
  }

  /**
   * All reduction. Produces true or 1, depending on the output type, if all of the elements in
   * the range are true or non-zero, otherwise produces a false or 0.
   * Null values are skipped.
   */
  public static SegmentedReductionAggregation all() {
    return new SegmentedReductionAggregation(Aggregation.all());
  }

  /**
   * Execute a reduction using a host-side user-defined function (UDF).
   * @param wrapper The wrapper for the native host UDF instance.
   * @return A new SegmentedReductionAggregation instance
   */
  public static SegmentedReductionAggregation hostUDF(HostUDFWrapper wrapper) {
    return new SegmentedReductionAggregation(Aggregation.hostUDF(wrapper));
  }
}
