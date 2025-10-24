/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
