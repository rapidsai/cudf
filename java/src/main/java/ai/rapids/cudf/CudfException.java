/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * Exception thrown by cudf itself.
 */
public class CudfException extends RuntimeException {
  CudfException(String message) {
    this(message, "No native stacktrace is available.");
  }

  CudfException(String message, String nativeStacktrace) {
    super(message);
    this.nativeStacktrace = nativeStacktrace;
  }

  CudfException(String message, String nativeStacktrace, Throwable cause) {
    super(message, cause);
    this.nativeStacktrace = nativeStacktrace;
  }

  public final String getNativeStacktrace() {
    return nativeStacktrace;
  }

  private final String nativeStacktrace;
}
