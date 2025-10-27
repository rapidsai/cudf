/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Scan operation type.
 */
public enum ScanType {
  /**
   * Include the current row in the scan.
   */
  INCLUSIVE(true),
  /**
   * Exclude the current row from the scan.
   */
  EXCLUSIVE(false);

  ScanType(boolean isInclusive) {
    this.isInclusive = isInclusive;
  }

  final boolean isInclusive;
}
