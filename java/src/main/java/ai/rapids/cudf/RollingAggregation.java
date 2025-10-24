/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * An aggregation that can be used on rolling windows.
 */
public final class RollingAggregation {
  private final Aggregation wrapped;

  private RollingAggregation(Aggregation wrapped) {
    this.wrapped = wrapped;
  }

  long createNativeInstance() {
    return wrapped.createNativeInstance();
  }

  long getDefaultOutput() {
    return wrapped.getDefaultOutput();
  }

  /**
   * Add a column to the Aggregation so it can be used on a specific column of data.
   * @param columnIndex the index of the column to operate on.
   */
  public RollingAggregationOnColumn onColumn(int columnIndex) {
    return new RollingAggregationOnColumn(this, columnIndex);
  }

  @Override
  public int hashCode() {
    return wrapped.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    } else if (other instanceof RollingAggregation) {
      RollingAggregation o = (RollingAggregation) other;
      return wrapped.equals(o.wrapped);
    }
    return false;
  }

  /**
   * Rolling Window Sum
   */
  public static RollingAggregation sum() {
    return new RollingAggregation(Aggregation.sum());
  }


  /**
   * Rolling Window Min
   */
  public static RollingAggregation min() {
    return new RollingAggregation(Aggregation.min());
  }

  /**
   * Rolling Window Max
   */
  public static RollingAggregation max() {
    return new RollingAggregation(Aggregation.max());
  }

  /**
   * Rolling Window Standard Deviation with 1 as delta degrees of freedom(DDOF).
   */
  public static RollingAggregation standardDeviation() {
    return new RollingAggregation(Aggregation.standardDeviation());
  }

  /**
   * Rolling Window Standard Deviation with configurable delta degrees of freedom(DDOF).
   */
  public static RollingAggregation standardDeviation(int ddof) {
    return new RollingAggregation(Aggregation.standardDeviation(ddof));
  }

  /**
   * Count number of valid, a.k.a. non-null, elements.
   */
  public static RollingAggregation count() {
    return new RollingAggregation(Aggregation.count());
  }

  /**
   * Count number of elements.
   * @param nullPolicy INCLUDE if nulls should be counted. EXCLUDE if only non-null values
   *                   should be counted.
   */
  public static RollingAggregation count(NullPolicy nullPolicy) {
    return new RollingAggregation(Aggregation.count(nullPolicy));
  }

  /**
   * Arithmetic Mean
   */
  public static RollingAggregation mean() {
    return new RollingAggregation(Aggregation.mean());
  }


  /**
   * Index of max element.
   */
  public static RollingAggregation argMax() {
    return new RollingAggregation(Aggregation.argMax());
  }

  /**
   * Index of min element.
   */
  public static RollingAggregation argMin() {
    return new RollingAggregation(Aggregation.argMin());
  }


  /**
   * Get the row number.
   */
  public static RollingAggregation rowNumber() {
    return new RollingAggregation(Aggregation.rowNumber());
  }


  /**
   * In a rolling window return the value offset entries ahead or null if it is outside of the
   * window.
   */
  public static RollingAggregation lead(int offset) {
    return lead(offset, null);
  }

  /**
   * In a rolling window return the value offset entries ahead or the corresponding value from
   * defaultOutput if it is outside of the window. Note that this does not take any ownership of
   * defaultOutput and the caller mush ensure that defaultOutput remains valid during the life
   * time of this aggregation operation.
   */
  public static RollingAggregation lead(int offset, ColumnVector defaultOutput) {
    return new RollingAggregation(Aggregation.lead(offset, defaultOutput));
  }



  /**
   * In a rolling window return the value offset entries behind or null if it is outside of the
   * window.
   */
  public static RollingAggregation lag(int offset) {
    return lag(offset, null);
  }

  /**
   * In a rolling window return the value offset entries behind or the corresponding value from
   * defaultOutput if it is outside of the window. Note that this does not take any ownership of
   * defaultOutput and the caller mush ensure that defaultOutput remains valid during the life
   * time of this aggregation operation.
   */
  public static RollingAggregation lag(int offset, ColumnVector defaultOutput) {
    return new RollingAggregation(Aggregation.lag(offset, defaultOutput));
  }


  /**
   * Collect the values into a list. Nulls will be skipped.
   */
  public static RollingAggregation collectList() {
    return new RollingAggregation(Aggregation.collectList());
  }

  /**
   * Collect the values into a list.
   *
   * @param nullPolicy Indicates whether to include/exclude nulls during collection.
   */
  public static RollingAggregation collectList(NullPolicy nullPolicy) {
    return new RollingAggregation(Aggregation.collectList(nullPolicy));
  }


  /**
   * Collect the values into a set. All null values will be excluded, and all nan values are regarded as
   * unique instances.
   */
  public static RollingAggregation collectSet() {
    return new RollingAggregation(Aggregation.collectSet());
  }

  /**
   * Collect the values into a set.
   *
   * @param nullPolicy   Indicates whether to include/exclude nulls during collection.
   * @param nullEquality Flag to specify whether null entries within each list should be considered equal.
   * @param nanEquality  Flag to specify whether NaN values in floating point column should be considered equal.
   */
  public static RollingAggregation collectSet(NullPolicy nullPolicy, NullEquality nullEquality, NaNEquality nanEquality) {
    return new RollingAggregation(Aggregation.collectSet(nullPolicy, nullEquality, nanEquality));
  }

  /**
   * Select the nth element from a specified window.
   *
   * @param n          Indicates the index of the element to be selected from the window
   * @param nullPolicy Indicates whether null elements are to be skipped, or not
   */
  public static RollingAggregation nth(int n, NullPolicy nullPolicy) {
    return new RollingAggregation(Aggregation.nth(n, nullPolicy));
  }
}
