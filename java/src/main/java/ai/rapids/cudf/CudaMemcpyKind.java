/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
