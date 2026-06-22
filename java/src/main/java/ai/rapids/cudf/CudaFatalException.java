/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * CudaFatalException is a kind of CudaException which leaves the process in an inconsistent state
 * and any further CUDA work will return the same error.
 * To continue using CUDA, the process must be terminated and relaunched.
 */
public class CudaFatalException extends CudaException {
  CudaFatalException(String message, int errorCode) {
    this(message, "No native stacktrace is available.", errorCode);
  }

  CudaFatalException(String message, String nativeStacktrace, int errorCode) {
    super(message, nativeStacktrace, errorCode);
  }

  CudaFatalException(String message, String nativeStacktrace, int errorCode, Throwable cause) {
    super(message, nativeStacktrace, errorCode, cause);
  }
}
