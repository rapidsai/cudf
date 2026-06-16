/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * This class supports start/end NVTX profiling ranges.
 *
 * Start/end:
 *
 * The constructor instantiates a new NVTX range and keeps a unique handle that comes back
 * from the NVTX api (nvtxRangeId). The handle is used to later close such a range. This type
 * of range does not have the same order-of-operation requirements that the push/pop ranges have:
 * the `NvtxUniqueRange` instance can be passed to other scopes, and even to other threads
 * for the eventual call to close.
 *
 * It can be used in the same try-with-resources way as push/pop, or interleaved with other
 * ranges, like so:
 *
 * <pre>
 *   NvtxUniqueRange a = new NvtxUniqueRange("a", NvtxColor.RED);
 *   NvtxUniqueRange b = new NvtxUniqueRange("b", NvtxColor.BLUE);
 *   a.close();
 *   b.close();
 * </pre>
 */
public class NvtxUniqueRange implements AutoCloseable {
  private static final boolean isEnabled = Boolean.getBoolean("ai.rapids.cudf.nvtx.enabled");

  // this is a nvtxRangeId_t in the C++ api side
  private final long nvtxRangeId;

  // true if this range is already closed
  private boolean closed;

  static {
    if (isEnabled) {
      NativeDepsLoader.loadNativeDeps();
    }
  }

  public NvtxUniqueRange(String name, NvtxColor color) {
    this(name, color.colorBits);
  }

  public NvtxUniqueRange(String name, int colorBits) {
    if (isEnabled) {
      nvtxRangeId = start(name, colorBits);
    } else {
      // following the implementation in nvtx3, the default value of 0
      // is given when NVTX is disabled
      nvtxRangeId = 0;
    }
  }

  @Override
  public synchronized void close() {
    if (closed) {
      throw new IllegalStateException(
          "Cannot call close on an already closed NvtxUniqueRange!");
    }
    closed = true;
    if (isEnabled) {
      end(this.nvtxRangeId);
    }
  }

  private native long start(String name, int colorBits);
  private native void end(long nvtxRangeId);
}
