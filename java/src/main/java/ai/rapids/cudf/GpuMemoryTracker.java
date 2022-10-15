/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

import java.util.Optional;

/**
 * This is a helper class to track the maximum amount of GPU memory outstanding
 * for the current thread (stream in PTDS). If free ocurrs while tracking, and the 
 * free is for memory that wasn't created in the scope, or it was created in a different
 * thread, it will be ignored.
 *
 * The constructor enables a new memory tracking scope and .close stops tracking, and collects
 * the result.
 *
 * If `ai.rapids.cudf.gpuMemoryTracking.enabled` is false (default), the result of 
 * `getMaxOutstanding` is an empty java Optional<long>.
 *
 * Usage:
 *
 * <pre>
 *   try (GpuMemoryTracker a = new GpuMemoryTracker()) {
 *     ...
 *     try (GpuMemoryTracker b = new GpuMemoryTracker()) {
 *       ...
 *       // bMaxMemory is the maximum memory used while b is not closed
 *       Optional<long> bMaxMemory = b.getMaxOutsanding();
 *     }
 *     ...
 *
 *     // aMaxMemory is the maximum memory used while a is not closed
 *     // which includes bMaxMemory.
 *     Optional<long> aMaxMemory = a.getMaxOutsanding();
 *   }
 * </pre>
 *
 * Instances should be associated with a single thread and should be at a fine
 * granularity. Tracking memory when there could be free of buffers created in different
 * streams will have undeserired results.
 */
public class GpuMemoryTracker implements AutoCloseable {
  private static final boolean isEnabled = 
    Boolean.getBoolean("ai.rapids.cudf.gpuMemoryTracking.enabled");

  private long maxOutstanding;

  static {
    if (isEnabled) {
      NativeDepsLoader.loadNativeDeps();
    }
  }

  public GpuMemoryTracker() {
    if (isEnabled) {
      Rmm.pushThreadMemoryTracker();
    }
  }

  @Override
  public void close() {
    if (isEnabled) {
      maxOutstanding = Rmm.popThreadMemoryTracker();
    }
  }

  public Optional<Long> getMaxOutstanding() {
    if (isEnabled) {
      return Optional.of(maxOutstanding);
    } else {
      return Optional.empty();
    }
  }
}
