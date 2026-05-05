/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for multi-input aggregations (min_by, max_by, etc.)
 */
public class MultiInputAggregationTest {

  /**
   * Test min_by aggregation using role-tagged columns instead of a struct wrapper.
   */
  @Test
  public void testMinBy() {
    try (ColumnVector keys = ColumnVector.fromInts(1, 1, 2, 2, 3);
         ColumnVector ordering = ColumnVector.fromInts(5, 3, 8, 6, 9);
         ColumnVector values = ColumnVector.fromStrings("a", "b", "c", "d", "e");
         Table input = new Table(keys, ordering, values)) {
      long minById = MultiInputIds.next();

      try (Table result = input.groupBy(0).aggregate(
              GroupByAggregation.orderingForMinBy(minById).onColumn(1),
              GroupByAggregation.valueForMinBy(minById).onColumn(2));
           Table sorted = result.orderBy(OrderByArg.asc(0));
           ColumnVector expectedKeys = ColumnVector.fromInts(1, 2, 3);
           ColumnVector expectedOrdering = ColumnVector.fromInts(3, 6, 9);
           ColumnVector expectedValues = ColumnVector.fromStrings("b", "d", "e")) {
        assertEquals(3, sorted.getNumberOfColumns());
        assertEquals(3, sorted.getRowCount());
        AssertUtils.assertColumnsAreEqual(expectedKeys, sorted.getColumn(0));
        AssertUtils.assertColumnsAreEqual(expectedOrdering, sorted.getColumn(1));
        AssertUtils.assertColumnsAreEqual(expectedValues, sorted.getColumn(2));
      }
    }
  }

  /**
   * Test min_by with unsorted keys and value role before ordering role.
   */
  @Test
  public void testMinByWithUnsortedKeysAndValueFirst() {
    try (ColumnVector keys = ColumnVector.fromInts(2, 1, 2, 1, 3, 2);
         ColumnVector ordering = ColumnVector.fromInts(8, 5, 6, 3, 9, 4);
         ColumnVector values = ColumnVector.fromStrings("c", "a", "d", "b", "e", "f");
         Table input = new Table(keys, ordering, values)) {
      long minById = MultiInputIds.next();

      try (Table result = input.groupBy(0).aggregate(
              GroupByAggregation.valueForMinBy(minById).onColumn(2),
              GroupByAggregation.orderingForMinBy(minById).onColumn(1));
           Table sorted = result.orderBy(OrderByArg.asc(0));
           ColumnVector expectedKeys = ColumnVector.fromInts(1, 2, 3);
           ColumnVector expectedValues = ColumnVector.fromStrings("b", "f", "e");
           ColumnVector expectedOrdering = ColumnVector.fromInts(3, 4, 9)) {
        assertEquals(3, sorted.getNumberOfColumns());
        assertEquals(3, sorted.getRowCount());
        AssertUtils.assertColumnsAreEqual(expectedKeys, sorted.getColumn(0));
        AssertUtils.assertColumnsAreEqual(expectedValues, sorted.getColumn(1));
        AssertUtils.assertColumnsAreEqual(expectedOrdering, sorted.getColumn(2));
      }
    }
  }

  /**
   * Test that multiple min_by operations with different correlation IDs work in one groupby.
   */
  @Test
  public void testMultipleMinByOperations() {
    try (ColumnVector keys = ColumnVector.fromInts(1, 1, 2, 2);
         ColumnVector ordering1 = ColumnVector.fromInts(5, 3, 8, 6);
         ColumnVector values1 = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector ordering2 = ColumnVector.fromInts(10, 20, 15, 25);
         ColumnVector values2 = ColumnVector.fromStrings("w", "x", "y", "z");
         Table input = new Table(keys, ordering1, values1, ordering2, values2)) {
      long minById1 = MultiInputIds.next();
      long minById2 = MultiInputIds.next();

      try (Table result = input.groupBy(0).aggregate(
              GroupByAggregation.orderingForMinBy(minById1).onColumn(1),
              GroupByAggregation.valueForMinBy(minById1).onColumn(2),
              GroupByAggregation.orderingForMinBy(minById2).onColumn(3),
              GroupByAggregation.valueForMinBy(minById2).onColumn(4));
           Table sorted = result.orderBy(OrderByArg.asc(0));
           ColumnVector expectedKeys = ColumnVector.fromInts(1, 2);
           ColumnVector expectedOrdering1 = ColumnVector.fromInts(3, 6);
           ColumnVector expectedValues1 = ColumnVector.fromStrings("b", "d");
           ColumnVector expectedOrdering2 = ColumnVector.fromInts(10, 15);
           ColumnVector expectedValues2 = ColumnVector.fromStrings("w", "y")) {
        assertEquals(5, sorted.getNumberOfColumns());
        assertEquals(2, sorted.getRowCount());
        AssertUtils.assertColumnsAreEqual(expectedKeys, sorted.getColumn(0));
        AssertUtils.assertColumnsAreEqual(expectedOrdering1, sorted.getColumn(1));
        AssertUtils.assertColumnsAreEqual(expectedValues1, sorted.getColumn(2));
        AssertUtils.assertColumnsAreEqual(expectedOrdering2, sorted.getColumn(3));
        AssertUtils.assertColumnsAreEqual(expectedValues2, sorted.getColumn(4));
      }
    }
  }

  /**
   * Test result placement when multi-input aggregations are mixed with regular aggregations.
   */
  @Test
  public void testMinByWithRegularAggregations() {
    try (ColumnVector keys = ColumnVector.fromInts(1, 1, 2, 2);
         ColumnVector ordering = ColumnVector.fromInts(5, 3, 8, 6);
         ColumnVector values = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector amounts = ColumnVector.fromInts(10, 20, 30, 40);
         Table input = new Table(keys, ordering, values, amounts)) {
      long minById = MultiInputIds.next();

      try (Table result = input.groupBy(0).aggregate(
              GroupByAggregation.sum().onColumn(3),
              GroupByAggregation.orderingForMinBy(minById).onColumn(1),
              GroupByAggregation.valueForMinBy(minById).onColumn(2),
              GroupByAggregation.max().onColumn(3));
           Table sorted = result.orderBy(OrderByArg.asc(0));
           ColumnVector expectedKeys = ColumnVector.fromInts(1, 2);
           ColumnVector expectedSums = ColumnVector.fromLongs(30, 70);
           ColumnVector expectedOrdering = ColumnVector.fromInts(3, 6);
           ColumnVector expectedValues = ColumnVector.fromStrings("b", "d");
           ColumnVector expectedMax = ColumnVector.fromInts(20, 40)) {
        assertEquals(5, sorted.getNumberOfColumns());
        assertEquals(2, sorted.getRowCount());
        AssertUtils.assertColumnsAreEqual(expectedKeys, sorted.getColumn(0));
        AssertUtils.assertColumnsAreEqual(expectedSums, sorted.getColumn(1));
        AssertUtils.assertColumnsAreEqual(expectedOrdering, sorted.getColumn(2));
        AssertUtils.assertColumnsAreEqual(expectedValues, sorted.getColumn(3));
        AssertUtils.assertColumnsAreEqual(expectedMax, sorted.getColumn(4));
      }
    }
  }

  /**
   * Test that correlation IDs are unique across multiple calls.
   */
  @Test
  public void testMultiInputIdsUniqueness() {
    long id1 = MultiInputIds.next();
    long id2 = MultiInputIds.next();
    long id3 = MultiInputIds.next();

    assertNotEquals(id1, id2);
    assertNotEquals(id2, id3);
    assertNotEquals(id1, id3);
    assertTrue(id2 > id1);
    assertTrue(id3 > id2);
  }

  /**
   * Test that role-tagged aggregations compare by both role and correlation ID.
   */
  @Test
  public void testMinByAggregationEquality() {
    long id1 = MultiInputIds.next();
    long id2 = MultiInputIds.next();

    assertEquals(GroupByAggregation.valueForMinBy(id1), GroupByAggregation.valueForMinBy(id1));
    assertEquals(GroupByAggregation.orderingForMinBy(id1),
        GroupByAggregation.orderingForMinBy(id1));
    assertNotEquals(GroupByAggregation.valueForMinBy(id1), GroupByAggregation.valueForMinBy(id2));
    assertNotEquals(GroupByAggregation.orderingForMinBy(id1),
        GroupByAggregation.orderingForMinBy(id2));
    assertNotEquals(GroupByAggregation.valueForMinBy(id1),
        GroupByAggregation.orderingForMinBy(id1));
  }
}
