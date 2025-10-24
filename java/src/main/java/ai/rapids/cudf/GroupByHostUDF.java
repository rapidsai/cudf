/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * A base class represents the native `groupby_host_udf` class.
 */
public abstract class GroupByHostUDF {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHostUDF = 0L;

  // Return the native instance of this Host UDF.
  // This is to support the "createUDFInstance" API in HostUDFWrapper.
  public final long getNativeInstance() {
    if (nativeHostUDF == 0L) {
      nativeHostUDF = createNativeInstance();
    }
    return nativeHostUDF;
  }

  //======================================================
  // Two callbacks should be implemented by children.
  //======================================================

  // Called from the native when no input rows.
  protected abstract ColumnVector getEmptyOutput();

  // Called from the native to perform the actual aggregate.
  protected abstract ColumnVector aggregate();

  //======================================================
  // Utils for children to access the grouped information
  //======================================================

  /**
   * Access the offsets separating groups.
   * <br/>
   * The range of group "i" is "[offsets[i], offsets[i+1])". e.g.
   * Given the rows {1,1,1,2,2}, the offsets is {0,3,5} for 2 groups.
   */
  protected final ColumnView getGroupOffsets() {
    return new ColumnView(getGroupOffsetsView(getNativeInstance()));
  }

  /**
   * Access the input values grouped according to the input keys for
   * which the values within each group maintain their original order.
   */
  protected final ColumnView getGroupedValues() {
    return new ColumnView(getGroupedValuesView(getNativeInstance()));
  }

  /** Access the number of groups (i.e., number of distinct keys). */
  protected final long getNumGroups() {
    return getNumGroups(getNativeInstance());
  }

  //======================================================
  // Native methods
  //======================================================
  // non-static to access the Java instance from the native.
  private native long createNativeInstance();
  private static native long getGroupOffsetsView(long nativeUDF);
  private static native long getGroupedValuesView(long nativeUDF);
  private static native long getNumGroups(long nativeUDF);
}
