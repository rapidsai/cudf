/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * This class will house only tests that need to explicitly set non-empty nulls
 */
public class ColumnViewNonEmptyNullsTest extends CudfTestBase {

  private static final HostMemoryAllocator hostMemoryAllocator = DefaultHostMemoryAllocator.get();

  @Test
  void testAndNullReconfigureNulls() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(0, 100, null, null, Integer.MIN_VALUE, null);
         ColumnVector v1 = ColumnVector.fromBoxedInts(0, 100, 1, 2, Integer.MIN_VALUE, null);
         ColumnVector intResult = v1.mergeAndSetValidity(BinaryOp.BITWISE_AND, v0);
         ColumnVector v2 = ColumnVector.fromStrings("0", "100", "1", "2", "MIN_VALUE", "3");
         ColumnVector v3 = v0.mergeAndSetValidity(BinaryOp.BITWISE_AND, v1, v2);
         ColumnVector stringResult = v2.mergeAndSetValidity(BinaryOp.BITWISE_AND, v0, v1);
         ColumnVector stringExpected = ColumnVector.fromStrings("0", "100", null, null, "MIN_VALUE", null);
         ColumnVector noMaskResult = v2.mergeAndSetValidity(BinaryOp.BITWISE_AND)) {
      assertColumnsAreEqual(v0, intResult);
      assertColumnsAreEqual(v0, v3);
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(v2, noMaskResult);
    }
  }

  @Test
  void testOrNullReconfigureNulls() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(0, 100, null, null, Integer.MIN_VALUE, null);
         ColumnVector v1 = ColumnVector.fromBoxedInts(0, 100, 1, 2, Integer.MIN_VALUE, null);
         ColumnVector v2 = ColumnVector.fromBoxedInts(0, 100, 1, 2, Integer.MIN_VALUE, Integer.MAX_VALUE);
         ColumnVector intResultV0 = v1.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0);
         ColumnVector intResultV0V1 = v1.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0, v1);
         ColumnVector intResultMulti = v1.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0, v0, v1, v1, v0, v1, v0);
         ColumnVector intResultv0v1v2 = v2.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0, v1, v2);
         ColumnVector v3 = ColumnVector.fromStrings("0", "100", "1", "2", "MIN_VALUE", "3");
         ColumnVector stringResult = v3.mergeAndSetValidity(BinaryOp.BITWISE_OR, v0, v1);
         ColumnVector stringExpected = ColumnVector.fromStrings("0", "100", "1", "2", "MIN_VALUE", null);
         ColumnVector noMaskResult = v3.mergeAndSetValidity(BinaryOp.BITWISE_OR)) {
      assertColumnsAreEqual(v0, intResultV0);
      assertColumnsAreEqual(v1, intResultV0V1);
      assertColumnsAreEqual(v1, intResultMulti);
      assertColumnsAreEqual(v2, intResultv0v1v2);
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(v3, noMaskResult);
    }
  }

  /**
   * The caller needs to make sure to close the returned ColumnView
   */
  private ColumnView[] getColumnViewWithNonEmptyNulls() {
    List<Integer> list0 = Arrays.asList(1, 2, 3);
    List<Integer> list1 = Arrays.asList(4, 5, null);
    List<Integer> list2 = Arrays.asList(7, 8, 9);
    List<Integer> list3 = null;
    ColumnVector input = ColumnVectorTest.makeListsColumn(DType.INT32, list0, list1, list2, list3);
    // Modify the validity buffer
    BaseDeviceMemoryBuffer dmb = input.getDeviceBufferFor(BufferType.VALIDITY);
    try (HostMemoryBuffer newValidity = hostMemoryAllocator.allocate(64)) {
      newValidity.copyFromDeviceBuffer(dmb);
      BitVectorHelper.setNullAt(newValidity, 1);
      dmb.copyFromHostBuffer(newValidity);
    }
    try (HostColumnVector hostColumnVector = input.copyToHost()) {
      assert (hostColumnVector.isNull(1));
      assert (hostColumnVector.isNull(3));
    }
    try (ColumnVector expectedOffsetsBeforePurge = ColumnVector.fromInts(0, 3, 6, 9, 9)) {
      ColumnView offsetsCvBeforePurge = input.getListOffsetsView();
      assertColumnsAreEqual(expectedOffsetsBeforePurge, offsetsCvBeforePurge);
    }
    ColumnView colWithNonEmptyNulls = new ColumnView(input.type, input.rows, Optional.of(2L), dmb,
        input.getDeviceBufferFor(BufferType.OFFSET), input.getChildColumnViews());
    assertEquals(2, colWithNonEmptyNulls.nullCount);
    return new ColumnView[]{input, colWithNonEmptyNulls};
  }

  @Test
  void testPurgeNonEmptyNullsList() {
    ColumnView[] values = getColumnViewWithNonEmptyNulls();
    try (ColumnView colWithNonEmptyNulls = values[1];
         ColumnView input = values[0];
         // purge non-empty nulls
         ColumnView colWithEmptyNulls = colWithNonEmptyNulls.purgeNonEmptyNulls();
         ColumnVector expectedOffsetsAfterPurge = ColumnVector.fromInts(0, 3, 3, 6, 6);
         ColumnView offsetsCvAfterPurge = colWithEmptyNulls.getListOffsetsView()) {
      assertTrue(colWithNonEmptyNulls.hasNonEmptyNulls());
      assertColumnsAreEqual(expectedOffsetsAfterPurge, offsetsCvAfterPurge);
      assertFalse(colWithEmptyNulls.hasNonEmptyNulls());
    }
  }

  @Test
  void testPurgeNonEmptyNullsStruct() {
    ColumnView[] values = getColumnViewWithNonEmptyNulls();
    try (ColumnView listCol = values[1];
         ColumnView input = values[0];
         ColumnView stringsCol = ColumnVector.fromStrings("A", "col", "of", "Strings");
         ColumnView structView = ColumnView.makeStructView(stringsCol, listCol);
         ColumnView structWithEmptyNulls = structView.purgeNonEmptyNulls();
         ColumnView newListChild = structWithEmptyNulls.getChildColumnView(1);
         ColumnVector expectedOffsetsAfterPurge = ColumnVector.fromInts(0, 3, 3, 6, 6);
         ColumnView offsetsCvAfterPurge = newListChild.getListOffsetsView()) {
      assertColumnsAreEqual(expectedOffsetsAfterPurge, offsetsCvAfterPurge);
      assertFalse(newListChild.hasNonEmptyNulls());
    }
  }
}
