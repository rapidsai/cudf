/*
 * SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

enum CudaMemcpyKind {
  HOST_TO_HOST(0),     /*< Host   -> Host */
  HOST_TO_DEVICE(1),   /*< Host   -> Device */
  DEVICE_TO_HOST(2),   /*< Device -> Host */
  DEVICE_TO_DEVICE(3), /*< Device -> Device */
  DEFAULT(4);   /*< Direction of the transfer is inferred from the pointer values. Requires
                          unified virtual addressing */

  private final int value;

  CudaMemcpyKind(int value) {
    this.value = value;
  }

  int getValue() {
    return value;
  }
}
