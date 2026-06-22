/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Capture groups setting, closely following cudf::strings::capture_groups.
 *
 * For processing a regex pattern containing capture groups. These can be used
 * to optimize the generated regex instructions where the capture groups do not
 * require extracting the groups.
 */
public enum CaptureGroups {
  EXTRACT(0),     // capture groups processed normally for extract
  NON_CAPTURE(1); // convert all capture groups to non-capture groups

  final int nativeId; // Native id, for use with libcudf.
  private CaptureGroups(int nativeId) { // Only constant values should be used
    this.nativeId = nativeId;
  }
}
