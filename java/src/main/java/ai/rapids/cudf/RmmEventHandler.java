/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

public interface RmmEventHandler {
  /**
   * Invoked on a memory allocation failure.
   * @param sizeRequested number of bytes that failed to allocate
   * @return true if the memory allocation should be retried or false if it should fail
   */
  boolean onAllocFailure(long sizeRequested);

  /**
   * Get the memory thresholds that will trigger {@link #onAllocThreshold(long)}
   * to be called when one or more of the thresholds is crossed during a memory allocation.
   * A threshold is crossed when the total memory allocated before the RMM allocate operation
   * is less than a threshold value and the threshold value is less than or equal to the
   * total memory allocated after the RMM memory allocate operation.
   * @return allocate memory thresholds or null for no thresholds.
   */
  long[] getAllocThresholds();

  /**
   * Get the memory thresholds that will trigger {@link #onDeallocThreshold(long)}
   * to be called when one or more of the thresholds is crossed during a memory deallocation.
   * A threshold is crossed when the total memory allocated before the RMM deallocate operation
   * is greater than or equal to a threshold value and the threshold value is greater than the
   * total memory allocated after the RMM memory deallocate operation.
   * @return deallocate memory thresholds or null for no thresholds.
   */
  long[] getDeallocThresholds();

  /**
   * Invoked after an RMM memory allocate operation when an allocate threshold is crossed.
   * See {@link #getAllocThresholds()} for details on allocate threshold crossing.
   * <p>NOTE: Any exception thrown by this method will cause the corresponding allocation
   * that triggered the threshold callback to be released before the exception is
   * propagated to the application.
   * @param totalAllocSize total amount of memory allocated after the crossing
   */
  void onAllocThreshold(long totalAllocSize);

  /**
   * Invoked after an RMM memory deallocation operation when a deallocate threshold is crossed.
   * See {@link #getDeallocThresholds()} for details on deallocate threshold crossing.
   * <p>NOTE: Any exception thrown by this method will be propagated to the application
   * after the resource that triggered the threshold was released.
   * @param totalAllocSize total amount of memory allocated after the crossing
   */
  void onDeallocThreshold(long totalAllocSize);
}
