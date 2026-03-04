/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */
package ai.rapids.cudf;

/**
 * Used for the dropListDuplicates function
 * Specifies which duplicate to keep
 * @see cudf::duplicate_keep_option in /cpp/include/cudf/stream_compaction.hpp,
 * from which this enum is based off of. Values should be kept in sync.
 */
public enum DuplicateKeepOption {
  KEEP_ANY(0),    // keep any instance of a value
  KEEP_FIRST(1),  // only keep the first instance of an value
  KEEP_LAST(2),   // keep the last instance of an value
  KEEP_NONE(3);   // remove all instances of values with duplicates
  final int nativeId;

  DuplicateKeepOption(int nativeId) { this.nativeId = nativeId; }
}
