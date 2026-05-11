/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/** Utility methods for closing resources during exception handling. */
public final class Closeables {
  private Closeables() {}

  /**
   * Close {@code resource} on a cleanup path, attaching any close-time failure
   * as a suppressed exception on {@code primary} so the original cause is
   * preserved. No-op if {@code resource} is null.
   */
  public static void closeAndSuppress(AutoCloseable resource, Throwable primary) {
    if (resource == null) {
      return;
    }
    try {
      resource.close();
    } catch (Throwable closeFailure) {
      primary.addSuppressed(closeFailure);
    }
  }
}
