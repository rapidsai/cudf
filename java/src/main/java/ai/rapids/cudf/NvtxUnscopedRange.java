/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.cudf;

/**
 * Utility class to mark an NVTX profiling range.
 *
 * This class supports start/end NVTX ranges or "unscoped".
 *
 * Start/end:
 *
 * The constructor instantiates a new NVTX range and keeps a handle that comes back from the
 * NVTX api (nvtxRangeId) that used to later close such a range. This type of range does 
 * not have the same order-of-operation requirements that the push/pop ranges have: 
 * the `NvtxUnscopedRange` instance can be passed to other scopes, and even to other threads
 * for the eventual call to close.
 *
 * It can be used in the same try-with-resources way as push/pop, or interleaved with other
 * ranges, like so:
 *
 * <pre>
 *   NvtxUnscopedRange a = new NvtxUnscopedange("a", NvtxColor.RED);
 *   NvtxUnscopedRange b = new NvtxUnscopedange("b", NvtxColor.BLUE);
 *   a.close();
 *   b.close();
 * </pre>
 *
 * Start/end ranges are different in that they don't have the same correlation that the
 * push/pop ranges have. 
 */
public class NvtxUnscopedRange implements AutoCloseable {
  private static final boolean isEnabled = Boolean.getBoolean("ai.rapids.cudf.nvtx.enabled");

  // this is a nvtxRangeId_t in the C++ api side
  private long nvtxRangeId;

  // true if this range is already closed
  private boolean closed;

  static {
    if (isEnabled) {
      NativeDepsLoader.loadNativeDeps();
    }
  }

  public NvtxUnscopedRange(String name, NvtxColor color) {
    this(name, color.colorBits);
  }

  public NvtxUnscopedRange(String name, int colorBits) {
    if (isEnabled) {
      this.nvtxRangeId = start(name, colorBits);
    }
  }

  @Override
  public void close() {
    if (isEnabled) {
      if (closed) {
        throw new IllegalStateException(
          "Cannot call close on an already closed NvtxUnscopedRange!");
      }
      closed = true;
      end(this.nvtxRangeId);
    }
  }

  private native long start(String name, int colorBits);
  private native void end(long nvtxRangeId);
}
