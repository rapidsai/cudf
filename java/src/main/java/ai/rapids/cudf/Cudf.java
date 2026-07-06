/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

public class Cudf {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Initialize the libcudf JIT runtime and program cache.
   * This method may be called repeatedly. It validates runtime dependencies but does not actively
   * enable JIT evaluation; {@code CompiledExpression.computeColumn} continues to use the
   * process-level libcudf configuration.
   *
   * @throws CudfException if the JIT runtime cannot be initialized
   */
  public static native void initializeJitRuntime();

  /**
   * cuDF copies that are smaller than the threshold will use a kernel to copy, instead
   * of cudaMemcpyAsync.
   */
  public static native void setKernelPinnedCopyThreshold(long kernelPinnedCopyThreshold);

  /**
   * cudf allocations that are smaller than the threshold will use the pinned host
   * memory resource.
   */
  public static native void setPinnedAllocationThreshold(long pinnedAllocationThreshold);
}
