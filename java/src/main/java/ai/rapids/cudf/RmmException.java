/*
 * SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;


/**
 * Exception from RMM allocator.
 */
public class RmmException extends RuntimeException {
  RmmException(String message) {
    super(message);
  }

  RmmException(String message, Throwable cause) {
    super(message, cause);
  }
}
