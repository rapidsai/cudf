/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.nvcomp;

/** Base class for all nvcomp-specific exceptions */
public class NvcompException extends RuntimeException {
  NvcompException(String message) {
    super(message);
  }

  NvcompException(String message, Throwable cause) {
    super(message, cause);
  }
}
