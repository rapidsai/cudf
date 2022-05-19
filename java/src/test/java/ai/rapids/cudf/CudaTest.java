/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class CudaTest {

  @Test

  @Order(1)
  public void testGetCudaRuntimeInfo() {
    // The driver version is not necessarily larger than runtime version. Drivers of previous
    // version are also able to support runtime of later version, only if they support same
    // kinds of computeModes.
    assert Cuda.getDriverVersion() >= 1000;
    assert Cuda.getRuntimeVersion() >= 1000;
    assertEquals(Cuda.getNativeComputeMode(), Cuda.getComputeMode().nativeId);
  }

  @Test
  @Order(2)
  public void testCudaException() {
    assertThrows(CudaException.class, () -> {
          try {
            Cuda.memset(Long.MAX_VALUE, (byte) 0, 1024);
          } catch (CudaFatalException ignored) {
          } catch (CudaException ex) {
            assertEquals(CudaException.CudaError.cudaErrorInvalidValue, ex.cudaError);
            throw ex;
          }
        }
    );
    // non-fatal CUDA error will not fail subsequent CUDA calls
    try (ColumnVector cv = ColumnVector.fromBoxedInts(1, 2, 3, 4, 5)) {
    }
  }

  @Test
  @Order(3)
  public void testCudaFatalException() {
    try (ColumnView cv = ColumnView.fromDeviceBuffer(new BadDeviceBuffer(), 0, DType.INT8, 256);
         ColumnView ret = cv.sub(cv);
         HostColumnVector hcv = ret.copyToHost()) {
    } catch (CudaException ignored) {
    }

    // CUDA API invoked by libcudf failed because of previous unrecoverable fatal error
    assertThrows(CudaFatalException.class, () -> {
      try (ColumnView cv = ColumnView.fromDeviceBuffer(new BadDeviceBuffer(), 0, DType.INT8, 256);
           HostColumnVector hcv = cv.copyToHost()) {
      } catch (CudaFatalException ex) {
        assertEquals(CudaException.CudaError.cudaErrorIllegalAddress, ex.cudaError);
        throw ex;
      }
    });
  }

  @Test
  @Order(4)
  public void testCudaFatalExceptionFromRMM() {
    // CUDA API invoked by RMM failed because of previous unrecoverable fatal error
    assertThrows(CudaFatalException.class, () -> {
      try (ColumnVector cv = ColumnVector.fromBoxedInts(1, 2, 3, 4, 5)) {
      } catch (CudaFatalException ex) {
        assertEquals(CudaException.CudaError.cudaErrorIllegalAddress, ex.cudaError);
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
