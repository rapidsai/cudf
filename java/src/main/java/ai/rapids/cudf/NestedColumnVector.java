/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This class represents a nested column view that holds references to existing ColumnVector
 * children without copying data. This enables zero-copy creation of nested columns from existing
 * ColumnVector objects.
 * 
 * Currently only STRUCT type is supported. For LIST columns, use ColumnVector.makeListFromOffsets()
 * or other existing methods.
 * 
 * The class increments the reference count of child columns on construction and decrements
 * them on close, ensuring proper memory management.
 * 
 * Note: This class extends ColumnView rather than ColumnVector because ColumnVector is final.
 * However, you can convert this to a ColumnVector using copyToColumnVector() if needed.
 */
public final class NestedColumnVector extends ColumnView {

  private final List<ColumnVector> children;

  /**
   * Create a nested STRUCT column view from existing column vectors.
   * 
   * @param type the data type of the nested column, must be STRUCT
   * @param children the child columns for the STRUCT
   * @throws IllegalArgumentException if type is not STRUCT
   */
  public NestedColumnVector(DType type, ColumnVector... children) {
    super(createNestedColumnView(type, children));
    
    validateType(type);
    
    // Store children and increment their reference counts
    List<ColumnVector> childList = new ArrayList<>(children.length);
    for (ColumnVector child : children) {
      if (child == null) {
        throw new IllegalArgumentException("Child columns cannot be null");
      }
      child.incRefCount();
      childList.add(child);
    }
    this.children = Collections.unmodifiableList(childList);
  }

  /**
   * Validates that the type is supported.
   * 
   * @param type the data type to validate
   * @throws IllegalArgumentException if type is not STRUCT
   */
  private static void validateType(DType type) {
    if (!type.equals(DType.STRUCT)) {
      throw new IllegalArgumentException(
          "NestedColumnVector currently only supports STRUCT type. " +
          "For LIST columns, use ColumnVector.makeListFromOffsets() or other existing methods. " +
          "Got type: " + type);
    }
  }

  /**
   * Create a native column_view pointer from the children column vectors.
   * 
   * @param type the nested type (currently only STRUCT is supported)
   * @param children the child column vectors
   * @return a native pointer to the created cudf::column_view
   */
  private static long createNestedColumnView(DType type, ColumnVector... children) {
    validateType(type);
    
    if (children == null) {
      children = new ColumnVector[0];
    }
    
    long rows = 0;
    if (children.length > 0) {
      rows = children[0].getRowCount();
      // Verify all children have the same row count
      for (int i = 1; i < children.length; i++) {
        if (children[i].getRowCount() != rows) {
          throw new IllegalArgumentException(
              "All children must have the same row count. Expected: " + rows + 
              ", got: " + children[i].getRowCount() + " at index " + i);
        }
      }
    }
    
    long[] childHandles = new long[children.length];
    for (int i = 0; i < children.length; i++) {
      childHandles[i] = children[i].getNativeView();
    }
    
    return makeStructViewNative(childHandles, rows);
  }

  /**
   * Returns the native view handle for this nested column.
   * This returns a pointer to cudf::column_view that can be used in native operations.
   * 
   * @return the native pointer to the cudf::column_view
   */
  public long getNativeViewHandle() {
    return getNativeView();
  }

  /**
   * Close this nested column view and decrement reference counts of children.
   * Uses proper exception handling to ensure all children are closed even if
   * some operations fail.
   */
  @Override
  public void close() {
    // Close the parent view first
    super.close();
    
    // Decrement the reference counts of children
    // Collect exceptions to ensure all children are closed
    Throwable pending = null;
    for (ColumnVector child : children) {
      try {
        child.close();
      } catch (Throwable t) {
        if (pending == null) {
          pending = t;
        } else {
          pending.addSuppressed(t);
        }
      }
    }
    
    // Throw any collected exceptions after all cleanup is done
    if (pending != null) {
      if (pending instanceof RuntimeException) {
        throw (RuntimeException) pending;
      } else if (pending instanceof Error) {
        throw (Error) pending;
      } else {
        throw new RuntimeException(pending);
      }
    }
  }

  /**
   * Native method to create a STRUCT column view from child column views.
   * 
   * @param childHandles array of native pointers to child column_views
   * @param rowCount number of rows in the struct column
   * @return native pointer to the created cudf::column_view
   */
  private static native long makeStructViewNative(long[] childHandles, long rowCount);

  /**
   * Native method for LIST support (not currently implemented).
   * 
   * @param childHandles array containing a single native pointer to the child column_view
   * @param rowCount number of rows in the list column
   * @return native pointer to the created cudf::column_view
   */
  private static native long makeListViewNative(long[] childHandles, long rowCount);
}
