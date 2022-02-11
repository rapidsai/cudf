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
 * An aggregation that can be used for a grouped scan.
 */
public final class GroupByScanAggregation {
  private final Aggregation wrapped;

  private GroupByScanAggregation(Aggregation wrapped) {
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

  /**
   * Add a column to the Aggregation so it can be used on a specific column of data.
   * @param columnIndex the index of the column to operate on.
   */
  public GroupByScanAggregationOnColumn onColumn(int columnIndex) {
    return new GroupByScanAggregationOnColumn(this, columnIndex);
  }

  @Override
  public int hashCode() {
    return wrapped.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    } else if (other instanceof GroupByScanAggregation) {
      GroupByScanAggregation o = (GroupByScanAggregation) other;
      return wrapped.equals(o.wrapped);
    }
    return false;
  }

  /**
   * Sum Aggregation
   */
  public static GroupByScanAggregation sum() {
    return new GroupByScanAggregation(Aggregation.sum());
  }


  /**
   * Product Aggregation.
   */
  public static GroupByScanAggregation product() {
    return new GroupByScanAggregation(Aggregation.product());
  }

  /**
   * Min Aggregation
   */
  public static GroupByScanAggregation min() {
    return new GroupByScanAggregation(Aggregation.min());
  }

  /**
   * Max Aggregation
   */
  public static GroupByScanAggregation max() {
    return new GroupByScanAggregation(Aggregation.max());
  }

  /**
   * Count number of elements.
   * @param nullPolicy INCLUDE if nulls should be counted. EXCLUDE if only non-null values
   *                   should be counted.
   */
  public static GroupByScanAggregation count(NullPolicy nullPolicy) {
    return new GroupByScanAggregation(Aggregation.count(nullPolicy));
  }

  /**
   * Get the row's ranking.
   */
  public static GroupByScanAggregation rank() {
    return new GroupByScanAggregation(Aggregation.rank());
  }

  /**
   * Get the row's dense ranking.
   */
  public static GroupByScanAggregation denseRank() {
    return new GroupByScanAggregation(Aggregation.denseRank());
  }

  /**
   * Get the row's percent ranking.
   */
  public static GroupByScanAggregation percentRank() {
    return new GroupByScanAggregation(Aggregation.percentRank());
  }
}
