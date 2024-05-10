/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

public class CudaTest {

  @Test
  public void testGetCudaRuntimeInfo() {
    // The driver version is not necessarily larger than runtime version. Drivers of previous
    // version are also able to support runtime of later version, only if they support same
    // kinds of computeModes.
    assert Cuda.getDriverVersion() >= 1000;
    assert Cuda.getRuntimeVersion() >= 1000;
    assertEquals(Cuda.getNativeComputeMode(), Cuda.getComputeMode().nativeId);
  }

  @Test
  public void testCudaException() {
    assertThrows(CudaException.class, () -> {
          try {
            Cuda.freePinned(-1L);
          } catch (CudaFatalException fatalEx) {
            throw new AssertionError("Expected UnFatalError but got FatalError: " + fatalEx);
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
