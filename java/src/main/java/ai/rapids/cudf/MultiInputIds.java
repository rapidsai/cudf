/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Utility class for generating unique correlation IDs for multi-input aggregations.
 * These IDs are used to correlate multiple role-tagged aggregation instances that
 * belong to the same logical multi-input operation (e.g., min_by, max_by).
 */
public final class MultiInputIds {
  private static final AtomicLong COUNTER = new AtomicLong();

  /**
   * Generate the next unique multi-input correlation ID.
   * @return a unique long value to correlate multi-input aggregation roles
   */
  public static long next() {
    return COUNTER.incrementAndGet();
  }

  private MultiInputIds() {
  }
}
