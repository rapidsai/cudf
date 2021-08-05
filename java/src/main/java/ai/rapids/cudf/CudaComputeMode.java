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
 * This is the Java mapping of CUDA device compute modes.
 */
public enum CudaComputeMode {
  /**
   * Default compute mode
   * Multiple threads can use cudaSetDevice() with this device.
   */
  DEFAULT(0),
  /**
   * Compute-exclusive-thread mode
   * Only one thread in one process will be able to use cudaSetDevice() with this device.
   *
   * WARNING: This mode was deprecated! Using EXCLUSIVE_PROCESS instead.
   */
  EXCLUSIVE(1),
  /**
   * Compute-prohibited mode
   * No threads can use cudaSetDevice() with this device.
   */
  PROHIBITED(2),
  /**
   * Compute-exclusive-process mode
   * Many threads in one process will be able to use cudaSetDevice() with this device.
   */
  EXCLUSIVE_PROCESS(3);

  private CudaComputeMode(int nativeId) {
    this.nativeId = nativeId;
  }

  static CudaComputeMode fromNative(int nativeId) {
    for (CudaComputeMode mode : COMPUTE_MODES) {
      if (mode.nativeId == nativeId) return mode;
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a CudaComputeMode");
  }

  // mapping to the value of native mode
  final int nativeId;

  private static final CudaComputeMode[] COMPUTE_MODES = CudaComputeMode.values();
}
