/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
