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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class RmmMemoryAccessorTest extends CudfTestBase {
  @Test
  public void log() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, true, 1024*1024*1024);
    long address = Rmm.alloc(10, 0);
    try {
      assertNotEquals(0, address);
    } finally {
      Rmm.free(address, 0);
    }
    String log = Rmm.getLog();
    System.err.println(log);
    assertNotNull(log);
  }

  @Test
  public void init() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
    assertFalse(Rmm.isInitialized());
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, true, -1);
    assertTrue(Rmm.isInitialized());
    Rmm.shutdown();
    assertFalse(Rmm.isInitialized());
  }

  @Test
  public void allocate() {
    long address = Rmm.alloc(10, 0);
    try {
      assertNotEquals(0, address);
    } finally {
      Rmm.free(address, 0);
    }
  }

  @Test
  public void doubleInitFails() {
    if (!Rmm.isInitialized()) {
      Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, false, 0);
    }
    assertThrows(IllegalStateException.class, () -> {
      Rmm.initialize(RmmAllocationMode.POOL, false, 1024 * 1024);
    });
  }
}
