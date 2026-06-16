/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CudaTest {

  @Test
  public void testGetCudaRuntimeInfo() {
    // The driver version is not necessarily larger than runtime version. Drivers of previous
    // version are also able to support runtime of later version, only if they support same
    // kinds of computeModes.
    assert Cuda.getDriverVersion() >= 1000;
    assert Cuda.getRuntimeVersion() >= 1000;
    assertEquals(Cuda.getNativeComputeMode(), Cuda.getComputeMode().nativeId);

    // test UUID
    byte[] uuid = Cuda.getGpuUuid();
    assertEquals(uuid.length, 16);
    long v = 0;
    for (int i = 0; i < uuid.length; i++) {
      v += uuid[i];
    }
    assertNotEquals(0, v);
  }

  @Tag("noSanitizer")
  @Test
  public void testCudaException() {
    assertThrows(CudaException.class, () -> {
          try {
            Cuda.freePinned(-1L);
          } catch (CudaFatalException fatalEx) {
            throw new AssertionError("Expected CudaException but got fatal error", fatalEx);
          } catch (CudaException ex) {
            assertEquals(CudaException.CudaError.cudaErrorInvalidValue, ex.getCudaError());
            throw ex;
          }
        }
    );
    // non-fatal CUDA error will not fail subsequent CUDA calls
    try (ColumnVector cv = ColumnVector.fromBoxedInts(1, 2, 3, 4, 5)) {
    }
  }

}
