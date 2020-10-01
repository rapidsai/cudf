/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

public enum PadSide {
    LEFT(0),
    RIGHT(1),
    BOTH(2);

    private static final PadSide[] PAD_SIDES = PadSide.values();
    final int nativeId;

    PadSide(int nativeId) {
        this.nativeId = nativeId;
    }

    public int getNativeId() {
        return nativeId;
    }

    public static PadSide fromNative(int nativeId) {
        for (PadSide type : PAD_SIDES) {
            if (type.nativeId == nativeId) {
                return type;
            }
        }
        throw new IllegalArgumentException("Could not translate " + nativeId + " into a PadSide");
    }
}
