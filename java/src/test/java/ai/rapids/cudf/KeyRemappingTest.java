/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Basic tests to validate JNI plumbing for KeyRemapping.
 * Comprehensive algorithm testing is done in C++ tests.
 */
public class KeyRemappingTest extends CudfTestBase {

  @Test
  void testBasicIntegerKeys() {
    // Build table with some duplicate keys: [1, 2, 3, 2, 1]
    try (Table buildTable = new Table.TestBuilder()
            .column(1, 2, 3, 2, 1)
            .build();
         KeyRemapping remap = new KeyRemapping(buildTable)) {

      // Verify counts
      assertEquals(3, remap.getDistinctCount());
      assertEquals(2, remap.getMaxDuplicateCount());

      // Verify we can remap and get non-negative IDs
      try (ColumnVector result = remap.remapBuildKeys(buildTable);
           HostColumnVector hostResult = result.copyToHost()) {
        assertEquals(5, hostResult.getRowCount());
        for (int i = 0; i < hostResult.getRowCount(); i++) {
          assertTrue(hostResult.getInt(i) >= 0, "Build IDs should be non-negative");
        }
      }
    }
  }

  @Test
  void testStringKeys() {
    try (Table buildTable = new Table.TestBuilder()
            .column("apple", "banana", "cherry", "banana")
            .build();
         KeyRemapping remap = new KeyRemapping(buildTable)) {

      assertEquals(3, remap.getDistinctCount());
      assertEquals(2, remap.getMaxDuplicateCount());

      try (ColumnVector result = remap.remapBuildKeys(buildTable);
           HostColumnVector hostResult = result.copyToHost()) {
        assertEquals(4, hostResult.getRowCount());
      }
    }
  }

  @Test
  void testProbeNotFound() {
    try (Table buildTable = new Table.TestBuilder()
            .column(1, 2, 3)
            .build();
         KeyRemapping remap = new KeyRemapping(buildTable);
         Table probeTable = new Table.TestBuilder()
            .column(4, 5)  // Keys not in build
            .build();
         ColumnVector result = remap.remapProbeKeys(probeTable);
         HostColumnVector hostResult = result.copyToHost()) {

      // All probe keys should be NOT_FOUND
      for (int i = 0; i < hostResult.getRowCount(); i++) {
        assertEquals(KeyRemapping.NOT_FOUND_SENTINEL, hostResult.getInt(i));
      }
    }
  }

  @Test
  void testNullEqualityEqual() {
    // With EQUAL, nulls are treated as equal and get valid IDs
    try (ColumnVector col = ColumnVector.fromBoxedInts(1, 2, null, 2);
         Table buildTable = new Table(col);
         KeyRemapping remap = new KeyRemapping(buildTable, NullEquality.EQUAL)) {

      // Distinct: 1, 2, null = 3
      assertEquals(3, remap.getDistinctCount());
      assertEquals(NullEquality.EQUAL, remap.getNullEquality());

      try (ColumnVector result = remap.remapBuildKeys(buildTable);
           HostColumnVector hostResult = result.copyToHost()) {
        // All IDs should be non-negative (including null row)
        for (int i = 0; i < hostResult.getRowCount(); i++) {
          assertTrue(hostResult.getInt(i) >= 0);
        }
      }
    }
  }

  @Test
  void testNullEqualityUnequal() {
    // With UNEQUAL, null rows get BUILD_NULL_SENTINEL
    try (ColumnVector col = ColumnVector.fromBoxedInts(1, 2, null, 2);
         Table buildTable = new Table(col);
         KeyRemapping remap = new KeyRemapping(buildTable, NullEquality.UNEQUAL)) {

      // Distinct: 1, 2 = 2 (null is skipped)
      assertEquals(2, remap.getDistinctCount());
      assertEquals(NullEquality.UNEQUAL, remap.getNullEquality());

      try (ColumnVector result = remap.remapBuildKeys(buildTable);
           HostColumnVector hostResult = result.copyToHost()) {
        // Null row (index 2) should have BUILD_NULL_SENTINEL
        assertEquals(KeyRemapping.BUILD_NULL_SENTINEL, hostResult.getInt(2));
        // Non-null rows should have non-negative IDs
        assertTrue(hostResult.getInt(0) >= 0);
        assertTrue(hostResult.getInt(1) >= 0);
        assertTrue(hostResult.getInt(3) >= 0);
      }
    }
  }

  @Test
  void testEmptyTable() {
    try (Table buildTable = new Table.TestBuilder()
            .column(new Integer[0])
            .build();
         KeyRemapping remap = new KeyRemapping(buildTable)) {

      assertEquals(0, remap.getDistinctCount());
      assertEquals(0, remap.getMaxDuplicateCount());
    }
  }

  @Test
  void testReleaseBuildKeys() {
    Table buildTable = new Table.TestBuilder()
            .column(1, 2, 3)
            .build();
    try (KeyRemapping remap = new KeyRemapping(buildTable)) {
      assertFalse(remap.isBuildKeysReleased());

      // Release the build keys
      Table released = remap.releaseBuildKeys();
      assertTrue(remap.isBuildKeysReleased());

      // Can still use remap
      assertEquals(3, remap.getDistinctCount());

      // Must close the released table ourselves
      released.close();
    }
    // Original buildTable is still ours to close
    buildTable.close();
  }
}
