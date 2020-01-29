/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package ai.rapids.cudf;

enum MaskState {
  UNALLOCATED(0),
  UNINITIALIZED(1),
  ALL_VALID(2),
  ALL_NULL(3);

  private static final MaskState[] MASK_STATES = MaskState.values();
  final int nativeId;

  MaskState(int nativeId) {
    this.nativeId = nativeId;
  }

  static MaskState fromNative(int nativeId) {
    for (MaskState type : MASK_STATES) {
      if (type.nativeId == nativeId) {
        return type;
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a MaskState");
  }
}
