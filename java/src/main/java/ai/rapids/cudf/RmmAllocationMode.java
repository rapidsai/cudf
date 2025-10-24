/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

public class RmmAllocationMode {
  /**
   * Use cudaMalloc for allocation
   */
  public static final int CUDA_DEFAULT = 0x00000000;
  /**
   * Use pool suballocation strategy
   */
  public static final int POOL = 0x00000001;
  /**
   * Use cudaMallocManaged rather than cudaMalloc
   */
  public static final int CUDA_MANAGED_MEMORY = 0x00000002;
  /**
   * Use arena suballocation strategy
   */
  public static final int ARENA = 0x00000004;
  /**
   * Use CUDA async suballocation strategy
   */
  public static final int CUDA_ASYNC = 0x00000008;
  /**
   * Use CUDA async suballocation strategy with fabric handles that are
   * peer accessible with read-write access
   */
  public static final int CUDA_ASYNC_FABRIC = 0x00000010;
}
