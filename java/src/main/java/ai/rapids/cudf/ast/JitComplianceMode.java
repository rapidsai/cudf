/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/**
 * Arithmetic compliance mode for JIT AST operations.
 * NOTE: This must be kept in sync with `jni_to_jit_compliance_mode` in CompiledExpression.cpp!
 */
public enum JitComplianceMode {
  DEFAULT(0),
  ANSI(1),
  ANSI_TRY(2);

  private final byte nativeId;

  JitComplianceMode(int nativeId) {
    this.nativeId = (byte) nativeId;
    assert this.nativeId == nativeId;
  }

  int getSerializedSize() {
    return Byte.BYTES;
  }

  void serialize(ByteBuffer bb) {
    bb.put(nativeId);
  }
}
