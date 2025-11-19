/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * Exception thrown when CUDF operation results in a column size
 * exceeding CUDF column size limits
 */
public class CudfColumnSizeOverflowException extends CudfException {
  CudfColumnSizeOverflowException(String message) {
    super(message);
  }

  CudfColumnSizeOverflowException(String message, String nativeStacktrace) {
    super(message, nativeStacktrace);
  }

  CudfColumnSizeOverflowException(String message, String nativeStacktrace, Throwable cause) {
    super(message, nativeStacktrace, cause);
  }
}
