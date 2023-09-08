/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
