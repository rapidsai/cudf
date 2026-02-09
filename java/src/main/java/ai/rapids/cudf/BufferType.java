/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Types of buffers supported by ColumnVectors and HostColumnVectors
 */
public enum BufferType {
  VALIDITY,
  OFFSET,
  DATA
}
