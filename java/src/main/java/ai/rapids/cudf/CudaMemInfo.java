/*
 * SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * Represent free and total device memory.
 */
public class CudaMemInfo {
  /**
   * free memory in bytes
   */
  public final long free;
  /**
   * total memory in bytes
   */
  public final long total;

  CudaMemInfo(long free, long total) {
    this.free = free;
    this.total = total;
  }
}
