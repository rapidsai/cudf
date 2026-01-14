/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
