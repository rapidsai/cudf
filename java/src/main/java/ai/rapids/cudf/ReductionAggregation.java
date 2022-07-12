/*
 *
 *  Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
public final class ReductionAggregation {
  private final Aggregation wrapped;

  private ReductionAggregation(Aggregation wrapped) {
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
    } else if (other instanceof ReductionAggregation) {
      ReductionAggregation o = (ReductionAggregation) other;
      return wrapped.equals(o.wrapped);
    }
    return false;
  }

  /**
   * Sum Aggregation
   */
  public static ReductionAggregation sum() {
    return new ReductionAggregation(Aggregation.sum());
  }

  /**
   * Product Aggregation.
   */
  public static ReductionAggregation product() {
    return new ReductionAggregation(Aggregation.product());
  }

  /**
   * Min Aggregation
   */
  public static ReductionAggregation min() {
    return new ReductionAggregation(Aggregation.min());
  }

  /**
   * Max Aggregation
   */
  public static ReductionAggregation max() {
    return new ReductionAggregation(Aggregation.max());
  }

  /**
   * Any reduction. Produces a true or 1, depending on the output type,
   * if any of the elements in the range are true or non-zero, otherwise produces a false or 0.
   * Null values are skipped.
   */
  public static ReductionAggregation any() {
    return new ReductionAggregation(Aggregation.any());
  }

  /**
   * All reduction. Produces true or 1, depending on the output type, if all of the elements in
   * the range are true or non-zero, otherwise produces a false or 0.
   * Null values are skipped.
   */
  public static ReductionAggregation all() {
    return new ReductionAggregation(Aggregation.all());
  }


  /**
   * Sum of squares reduction.
   */
  public static ReductionAggregation sumOfSquares() {
    return new ReductionAggregation(Aggregation.sumOfSquares());
  }

  /**
   * Arithmetic mean reduction.
   */
  public static ReductionAggregation mean() {
    return new ReductionAggregation(Aggregation.mean());
  }


  /**
   * Variance aggregation with 1 as the delta degrees of freedom.
   */
  public static ReductionAggregation variance() {
    return new ReductionAggregation(Aggregation.variance());
  }

  /**
   * Variance aggregation.
   * @param ddof delta degrees of freedom. The divisor used in calculation of variance is
   *             <code>N - ddof</code>, where N is the population size.
   */
  public static ReductionAggregation variance(int ddof) {
    return new ReductionAggregation(Aggregation.variance(ddof));
  }

  /**
   * Standard deviation aggregation with 1 as the delta degrees of freedom.
   */
  public static ReductionAggregation standardDeviation() {
    return new ReductionAggregation(Aggregation.standardDeviation());
  }

  /**
   * Standard deviation aggregation.
   * @param ddof delta degrees of freedom. The divisor used in calculation of std is
   *             <code>N - ddof</code>, where N is the population size.
   */
  public static ReductionAggregation standardDeviation(int ddof) {
    return new ReductionAggregation(Aggregation.standardDeviation(ddof));
  }


  /**
   * Median reduction.
   */
  public static ReductionAggregation median() {
    return new ReductionAggregation(Aggregation.median());
  }

  /**
   * Aggregate to compute the specified quantiles. Uses linear interpolation by default.
   */
  public static ReductionAggregation quantile(double ... quantiles) {
    return new ReductionAggregation(Aggregation.quantile(quantiles));
  }

  /**
   * Aggregate to compute various quantiles.
   */
  public static ReductionAggregation quantile(QuantileMethod method, double ... quantiles) {
    return new ReductionAggregation(Aggregation.quantile(method, quantiles));
  }


  /**
   * Number of unique, non-null, elements.
   */
  public static ReductionAggregation nunique() {
    return new ReductionAggregation(Aggregation.nunique());
  }

  /**
   * Number of unique elements.
   * @param nullPolicy INCLUDE if nulls should be counted else EXCLUDE. If nulls are counted they
   *                   compare as equal so multiple null values in a range would all only
   *                   increase the count by 1.
   */
  public static ReductionAggregation nunique(NullPolicy nullPolicy) {
    return new ReductionAggregation(Aggregation.nunique(nullPolicy));
  }

  /**
   * Get the nth, non-null, element in a group.
   * @param offset the offset to look at. Negative numbers go from the end of the group. Any
   *               value outside of the group range results in a null.
   */
  public static ReductionAggregation nth(int offset) {
    return new ReductionAggregation(Aggregation.nth(offset));
  }

  /**
   * Get the nth element in a group.
   * @param offset the offset to look at. Negative numbers go from the end of the group. Any
   *               value outside of the group range results in a null.
   * @param nullPolicy INCLUDE if nulls should be included in the aggregation or EXCLUDE if they
   *                   should be skipped.
   */
  public static ReductionAggregation nth(int offset, NullPolicy nullPolicy) {
    return new ReductionAggregation(Aggregation.nth(offset, nullPolicy));
  }

  /**
   * tDigest reduction.
   */
  public static ReductionAggregation createTDigest(int delta) {
    return new ReductionAggregation(Aggregation.createTDigest(delta));
  }

  /**
   * tDigest merge reduction.
   */
  public static ReductionAggregation mergeTDigest(int delta) {
    return new ReductionAggregation(Aggregation.mergeTDigest(delta));
  }

  /*
   * Collect the values into a list. Nulls will be skipped.
   */
  public static ReductionAggregation collectList() {
    return new ReductionAggregation(Aggregation.collectList());
  }

  /**
   * Collect the values into a list.
   *
   * @param nullPolicy Indicates whether to include/exclude nulls during collection.
   */
  public static ReductionAggregation collectList(NullPolicy nullPolicy) {
    return new ReductionAggregation(Aggregation.collectList(nullPolicy));
  }

  /**
   * Collect the values into a set. All null values will be excluded, and all NaN values are regarded as
   * unique instances.
   */
  public static ReductionAggregation collectSet() {
    return new ReductionAggregation(Aggregation.collectSet());
  }

  /**
   * Collect the values into a set.
   *
   * @param nullPolicy   Indicates whether to include/exclude nulls during collection.
   * @param nullEquality Flag to specify whether null entries within each list should be considered equal.
   * @param nanEquality  Flag to specify whether NaN values in floating point column should be considered equal.
   */
  public static ReductionAggregation collectSet(NullPolicy nullPolicy,
      NullEquality nullEquality, NaNEquality nanEquality) {
    return new ReductionAggregation(Aggregation.collectSet(nullPolicy, nullEquality, nanEquality));
  }

  /**
   * Merge the partial lists produced by multiple CollectListAggregations.
   * NOTICE: The partial lists to be merged should NOT include any null list element (but can include null list entries).
   */
  public static ReductionAggregation mergeLists() {
    return new ReductionAggregation(Aggregation.mergeLists());
  }

  /**
   * Merge the partial sets produced by multiple CollectSetAggregations. Each null/NaN value will be regarded as
   * a unique instance.
   */
  public static ReductionAggregation mergeSets() {
    return new ReductionAggregation(Aggregation.mergeSets());
  }

  /**
   * Merge the partial sets produced by multiple CollectSetAggregations.
   *
   * @param nullEquality Flag to specify whether null entries within each list should be considered equal.
   * @param nanEquality  Flag to specify whether NaN values in floating point column should be considered equal.
   */
  public static ReductionAggregation mergeSets(NullEquality nullEquality, NaNEquality nanEquality) {
    return new ReductionAggregation(Aggregation.mergeSets(nullEquality, nanEquality));
  }

}
