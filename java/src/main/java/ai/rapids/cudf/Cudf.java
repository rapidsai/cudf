/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

public class Cudf {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

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
