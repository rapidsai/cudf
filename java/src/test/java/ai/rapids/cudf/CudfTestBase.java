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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CudfTestBase {
  static final long RMM_POOL_SIZE_DEFAULT = 512 * 1024 * 1024;

  final int rmmAllocationMode;
  final long rmmPoolSize;

  CudfTestBase() {
    this(RmmAllocationMode.POOL, RMM_POOL_SIZE_DEFAULT);
  }

  CudfTestBase(int allocationMode, long poolSize) {
    this.rmmAllocationMode = allocationMode;
    this.rmmPoolSize = poolSize;
  }

  @BeforeEach
  void beforeEach() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    if (!Rmm.isInitialized()) {
      Rmm.initialize(rmmAllocationMode, false, rmmPoolSize);
    }
  }

  @AfterAll
  static void afterAll() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }
}
