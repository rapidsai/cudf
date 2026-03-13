/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class NestedColumnVectorTest extends CudfTestBase {

  @Test
  public void testCreateStructFromColumns() {
    try (ColumnVector col1 = ColumnVector.fromInts(1, 2, 3, 4, 5);
         ColumnVector col2 = ColumnVector.fromStrings("a", "b", "c", "d", "e");
         NestedColumnVector struct = new NestedColumnVector(DType.STRUCT, col1, col2)) {
      
      assertEquals(DType.STRUCT, struct.getType());
      assertEquals(5, struct.getRowCount());
      assertEquals(2, struct.getNumChildren());
      
      // Verify children are accessible
      try (ColumnView child0 = struct.getChildColumnView(0);
           ColumnView child1 = struct.getChildColumnView(1)) {
        assertEquals(DType.INT32, child0.getType());
        assertEquals(DType.STRING, child1.getType());
        assertEquals(5, child0.getRowCount());
        assertEquals(5, child1.getRowCount());
      }
    }
  }

  @Test
  public void testCreateEmptyStruct() {
    // Create a struct with no children
    try (NestedColumnVector struct = new NestedColumnVector(DType.STRUCT)) {
      assertEquals(DType.STRUCT, struct.getType());
      assertEquals(0, struct.getRowCount());
      assertEquals(0, struct.getNumChildren());
    }
  }

  @Test
  public void testReferenceCountingOnChildren() {
    ColumnVector col1 = ColumnVector.fromInts(1, 2, 3);
    ColumnVector col2 = ColumnVector.fromStrings("x", "y", "z");
    
    // Initial ref counts should be 1
    assertEquals(1, col1.getRefCount());
    assertEquals(1, col2.getRefCount());
    
    // Create nested column - ref counts should increase
    NestedColumnVector struct = new NestedColumnVector(DType.STRUCT, col1, col2);
    assertEquals(2, col1.getRefCount());
    assertEquals(2, col2.getRefCount());
    
    // Close nested column - ref counts should decrease
    struct.close();
    assertEquals(1, col1.getRefCount());
    assertEquals(1, col2.getRefCount());
    
    // Clean up
    col1.close();
    col2.close();
  }

  @Test
  public void testChildrenMustHaveSameRowCount() {
    try (ColumnVector col1 = ColumnVector.fromInts(1, 2, 3);
         ColumnVector col2 = ColumnVector.fromStrings("a", "b")) {
      
      assertThrows(IllegalArgumentException.class, () -> {
        new NestedColumnVector(DType.STRUCT, col1, col2);
      });
    }
  }

  @Test
  public void testNullChildrenNotAllowed() {
    try (ColumnVector col1 = ColumnVector.fromInts(1, 2, 3)) {
      assertThrows(IllegalArgumentException.class, () -> {
        new NestedColumnVector(DType.STRUCT, col1, null);
      });
    }
  }

  @Test
  public void testOnlyStructTypeSupported() {
    try (ColumnVector col = ColumnVector.fromInts(1, 2, 3)) {
      // Test that LIST type throws exception
      assertThrows(IllegalArgumentException.class, () -> {
        new NestedColumnVector(DType.LIST, col);
      });
      
      // Test that non-nested types throw exception
      assertThrows(IllegalArgumentException.class, () -> {
        new NestedColumnVector(DType.INT32, col);
      });
      
      assertThrows(IllegalArgumentException.class, () -> {
        new NestedColumnVector(DType.STRING, col);
      });
    }
  }

  @Test
  public void testGetNativeViewHandle() {
    try (ColumnVector col1 = ColumnVector.fromInts(1, 2, 3);
         ColumnVector col2 = ColumnVector.fromDoubles(1.5, 2.5, 3.5);
         NestedColumnVector struct = new NestedColumnVector(DType.STRUCT, col1, col2)) {
      
      long viewHandle = struct.getNativeViewHandle();
      assertNotEquals(0, viewHandle);
      
      // Verify it's the same as getNativeView()
      assertEquals(struct.getNativeView(), viewHandle);
    }
  }

  @Test
  public void testNestedStructsWithMultipleChildren() {
    try (ColumnVector intCol = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector longCol = ColumnVector.fromLongs(10L, 20L, 30L, 40L);
         ColumnVector doubleCol = ColumnVector.fromDoubles(1.1, 2.2, 3.3, 4.4);
         ColumnVector stringCol = ColumnVector.fromStrings("one", "two", "three", "four");
         NestedColumnVector struct = new NestedColumnVector(DType.STRUCT, 
             intCol, longCol, doubleCol, stringCol)) {
      
      assertEquals(DType.STRUCT, struct.getType());
      assertEquals(4, struct.getRowCount());
      assertEquals(4, struct.getNumChildren());
      
      ColumnView[] children = struct.getChildColumnViews();
      try {
        assertEquals(4, children.length);
        assertEquals(DType.INT32, children[0].getType());
        assertEquals(DType.INT64, children[1].getType());
        assertEquals(DType.FLOAT64, children[2].getType());
        assertEquals(DType.STRING, children[3].getType());
      } finally {
        for (ColumnView child : children) {
          if (child != null) {
            child.close();
          }
        }
      }
    }
  }

  @Test
  public void testZeroCopyBehavior() {
    // Create columns
    try (ColumnVector col1 = ColumnVector.fromInts(1, 2, 3);
         ColumnVector col2 = ColumnVector.fromStrings("a", "b", "c")) {
      
      long col1MemSize = col1.getDeviceMemorySize();
      long col2MemSize = col2.getDeviceMemorySize();
      
      // Create nested column - should not copy data
      try (NestedColumnVector struct = new NestedColumnVector(DType.STRUCT, col1, col2)) {
        // The struct itself has minimal overhead (just metadata)
        // It should not duplicate the children's data
        long structMemSize = struct.getDeviceMemorySize();
        
        // The struct's memory should be for metadata only, not duplicating child data
        // This is a rough check - exact size depends on implementation details
        assertTrue(structMemSize < col1MemSize + col2MemSize,
            "Struct should not copy child data");
      }
    }
  }

  @Test
  public void testConvertToColumnVector() {
    // Test that NestedColumnVector can be converted to ColumnVector
    try (ColumnVector col1 = ColumnVector.fromInts(1, 2, 3);
         ColumnVector col2 = ColumnVector.fromStrings("a", "b", "c");
         NestedColumnVector struct = new NestedColumnVector(DType.STRUCT, col1, col2);
         ColumnVector cv = struct.copyToColumnVector()) {
      
      assertEquals(DType.STRUCT, cv.getType());
      assertEquals(3, cv.getRowCount());
      assertEquals(2, cv.getNumChildren());
    }
  }

  @Test
  public void testUseInTableOperations() {
    // Test that NestedColumnVector can be converted and used in Table operations
    try (ColumnVector col1 = ColumnVector.fromInts(1, 2, 3);
         ColumnVector col2 = ColumnVector.fromStrings("a", "b", "c");
         NestedColumnVector struct = new NestedColumnVector(DType.STRUCT, col1, col2);
         ColumnVector cv = struct.copyToColumnVector();
         Table t = new Table(cv)) {
      
      assertEquals(1, t.getNumberOfColumns());
      assertEquals(3, t.getRowCount());
      
      try (ColumnVector resultCol = t.getColumn(0)) {
        assertEquals(DType.STRUCT, resultCol.getType());
        assertEquals(3, resultCol.getRowCount());
      }
    }
  }

  @Test
  public void testNestedColumnVectorWithNulls() {
    // Create columns with nulls
    try (ColumnVector col1 = ColumnVector.fromBoxedInts(1, null, 3, null, 5);
         ColumnVector col2 = ColumnVector.fromStrings("a", "b", null, "d", null);
         NestedColumnVector struct = new NestedColumnVector(DType.STRUCT, col1, col2)) {
      
      assertEquals(5, struct.getRowCount());
      assertEquals(2, struct.getNumChildren());
      
      // Verify children maintain their null masks
      try (ColumnView child0 = struct.getChildColumnView(0);
           ColumnView child1 = struct.getChildColumnView(1)) {
        assertEquals(2, child0.getNullCount());
        assertEquals(2, child1.getNullCount());
      }
    }
  }
}
