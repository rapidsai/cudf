/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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

/**
 * Provides a convenient way to setup a prediction for a pinned data operation. The prediction
 * should be used in a try-with-resources block so that close is guaranteed to be called and
 * end the prediction. Do not nest these. Make sure that the prediction is as close to the actual
 * memory allocation as possible to avoid the possibility of nesting. This will typically be right
 * around the JNI call that does the operation.
 */
final class HostPrediction implements AutoCloseable {
  private final String note;

  /**
   * Predict how much memory is going to be used.
   * @param amount the number of bytes to be used. Often this can be based off of the size of the
   *               input columns/tables.
   * @param note a string that is used to describe the operation. This is just for debugging and
   *             should probably be a static string so it does not require any new memory
   *             allocation or manipulation.
   */
  HostPrediction(long amount, String note) {
    this.note = note;
    MemoryListener.hostPrediction(amount, note);
  }

  @Override
  public void close() {
    MemoryListener.hostEndPrediction(note);
  }
}
