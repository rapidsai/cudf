/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.nvcomp;

/** Exception thrown from nvcomp indicating a CUDA error occurred. */
public class NvcompCudaException extends NvcompException {
  NvcompCudaException(String message) {
    super(message);
  }

  NvcompCudaException(String message, Throwable cause) {
    super(message, cause);
  }
}
