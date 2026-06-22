/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
