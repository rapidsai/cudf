/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Specify whether to include nulls or exclude nulls in an operation.
 */
public enum NullPolicy {
  EXCLUDE(false),
  INCLUDE(true);

  NullPolicy(boolean includeNulls) {
    this.includeNulls = includeNulls;
  }

  final boolean includeNulls;
}
