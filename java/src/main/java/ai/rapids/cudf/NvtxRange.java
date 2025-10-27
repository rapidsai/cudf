/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * This class supports push/pop NVTX profiling ranges, or "scoped" ranges.
 *
 * The constructor pushes an NVTX range and the close method pops off the most recent range that
 * was pushed. Therefore instances of this class should always be used in a try-with-resources
 * block to guarantee that ranges are always closed in the proper order. For example:
 * <pre>
 *   try (NvtxRange a = new NvtxRange("a", NvtxColor.RED)) {
 *     ...
 *     try (NvtxRange b = new NvtxRange("b", NvtxColor.BLUE)) {
 *       ...
 *     }
 *     ...
 *   }
 * </pre>
 *
 * Instances should be associated with a single thread to avoid pushing an NVTX range in
 * one thread and then trying to pop the range in a different thread.
 *
 * Push/pop ranges show a stacking behavior in tools such as Nsight, where newly pushed
 * ranges are correlated and enclosed by the prior pushed range (in the example above,
 * "b" is enclosed by "a").
 */
public class NvtxRange implements AutoCloseable {
  private static final boolean isEnabled = Boolean.getBoolean("ai.rapids.cudf.nvtx.enabled");

  static {
    if (isEnabled) {
      NativeDepsLoader.loadNativeDeps();
    }
  }

  public NvtxRange(String name, NvtxColor color) {
    this(name, color.colorBits);
  }

  public NvtxRange(String name, int colorBits) {
    if (isEnabled) {
      push(name, colorBits);
    }
  }

  public static void pushRange(String name, NvtxColor color) {
    if (isEnabled) {
      push(name, color.colorBits);
    }
  }

  public static void popRange() {
    if (isEnabled) {
      pop();
    }
  }

  @Override
  public void close() {
    if (isEnabled) {
      pop();
    }
  }

  private static native void push(String name, int colorBits);
  private static native void pop();
}
