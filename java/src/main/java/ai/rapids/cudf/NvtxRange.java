/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
 * This class supports two types of NVTX ranges: push/pop and start/end:
 *
 * Push/pop:
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
 *
 * Start/end:
 *
 * The constructor instantiates a new NVTX range and keeps a handle that comes back from the
 * NVTX api (nvtxRangeId) that used to later close such a range. This type of range does 
 * not have the same order-of-operation requirements that the push/pop ranges have: 
 * the `NvtxRange` instance can be passed to other scopes, and even to other threads 
 * for the eventual call to close.
 *
 * It can be used in the same try-with-resources way as push/pop, or interleaved with other
 * ranges, like so:
 *
 * <pre>
 *   NvtxRange a = new NvtxRange("a", NvtxColor.RED, NvtxRange.Type.STARTEND);
 *   NvtxRange b = new NvtxRange("b", NvtxColor.BLUE, NvtxRange.Type.STARTEND);
 *   a.close();
 *   b.close();
 * </pre>
 *
 * Start/end ranges are different in that they don't have the same correlation that the
 * push/pop ranges have. 
 */
public class NvtxRange implements AutoCloseable {
  public enum Type {
    PUSH,
    STARTEND
  }
  private Type type;

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

  public NvtxRange(String name, NvtxColor color) {
    this(name, color.colorBits, Type.PUSH);
  }

  public NvtxRange(String name, NvtxColor color, Type type) {
    this(name, color.colorBits, type);
  }

  public NvtxRange(String name, int colorBits, Type type) {
    this.type = type;
    if (isEnabled) {
      if (type == Type.PUSH) {
        push(name, colorBits);
      } else {
        this.nvtxRangeId = start(name, colorBits);
      } 
    }
  }

  @Override
  public void close() {
    if (isEnabled) {
      if (closed) {
        throw new IllegalStateException("Cannot call close on an already closed NvtxRange!");
      }
      closed = true;
      if (type == Type.PUSH) {
        pop();
      } else {
        end(this.nvtxRangeId);
      }
    }
  }

  private native void push(String name, int colorBits);
  private native void pop();
  private native long start(String name, int colorBits);
  private native void end(long nvtxRangeId);
}
