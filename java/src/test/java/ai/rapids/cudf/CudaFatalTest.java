/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CudaFatalTest {

  @Test
  public void testCudaFatalException() {
    try (ColumnVector cv = ColumnVector.fromInts(1, 2, 3, 4, 5)) {

      try (ColumnView badCv = ColumnView.fromDeviceBuffer(new BadDeviceBuffer(), 0, DType.INT8, 256);
           ColumnView ret = badCv.sub(badCv);
           HostColumnVector hcv = ret.copyToHost()) {
      } catch (CudaException ignored) {
      }

      // CUDA API invoked by libcudf failed because of previous unrecoverable fatal error
      assertThrows(CudaFatalException.class, () -> {
        try (ColumnVector cv2 = cv.asLongs()) {
        } catch (CudaFatalException ex) {
          assertEquals(CudaException.CudaError.cudaErrorIllegalAddress, ex.getCudaError());
          throw ex;
        }
      });
    }

    // CUDA API invoked by RMM failed because of previous unrecoverable fatal error
    assertThrows(CudaFatalException.class, () -> {
      try (ColumnVector cv = ColumnVector.fromBoxedInts(1, 2, 3, 4, 5)) {
      } catch (CudaFatalException ex) {
        assertEquals(CudaException.CudaError.cudaErrorIllegalAddress, ex.getCudaError());
        throw ex;
      }
    });
  }

  private static class BadDeviceBuffer extends BaseDeviceMemoryBuffer {
    public BadDeviceBuffer() {
      super(256L, 256L, (MemoryBufferCleaner) null);
    }

    @Override
    public MemoryBuffer slice(long offset, long len) {
      return null;
    }
  }

}
