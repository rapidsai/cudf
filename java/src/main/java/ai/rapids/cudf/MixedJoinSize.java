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

/** This class tracks size information associated with a mixed table join. */
public final class MixedJoinSize implements AutoCloseable {
  private final long outputRowCount;
  // This is in flux, avoid exposing publicly until the dust settles.
  private ColumnVector matches;

  MixedJoinSize(long outputRowCount, ColumnVector matches) {
    this.outputRowCount = outputRowCount;
    this.matches = matches;
  }

  /** Return the number of output rows that would be generated from the mixed join */
  public long getOutputRowCount() {
    return outputRowCount;
  }

  ColumnVector getMatches() {
    return matches;
  }

  @Override
  public synchronized void close() {
    matches.close();
  }
}
