/*
 *
 *  Copyright (c) 2022, NVIDIA CORPORATION.
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
}
