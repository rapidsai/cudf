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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.*;

class PinnedMemoryPoolTest extends CudfTestBase {
  private static final Logger log = LoggerFactory.getLogger(PinnedMemoryPoolTest.class);

  @AfterEach
  void teardown() {
    if (PinnedMemoryPool.isInitialized()) {
      PinnedMemoryPool.shutdown();
    }
  }

  @Test
  void init() {
    assertFalse(PinnedMemoryPool.isInitialized());
    PinnedMemoryPool.initialize(1024*1024*500L);
    assertTrue(PinnedMemoryPool.isInitialized());
    PinnedMemoryPool.shutdown();
    assertFalse(PinnedMemoryPool.isInitialized());
  }

  @Test
  void allocate() {
    PinnedMemoryPool.initialize(1024*1024*500L);
    for (int i = 2048000; i < 1024*1024*1024; i = i * 2) {
      log.warn("STARTING TEST FOR size = " + i);
      HostMemoryBuffer buff = null;
      HostMemoryBuffer buff2 = null;
      HostMemoryBuffer buff3 = null;
      try {
        buff = PinnedMemoryPool.allocate(i);
        assertEquals(i, buff.length);
        buff2 = PinnedMemoryPool.allocate(i / 2);
        assertEquals(i/2, buff2.length);
        buff.close();
        buff = null;
        buff3 = PinnedMemoryPool.allocate(i * 2);
        assertEquals(i * 2, buff3.length);
      } finally {
        if (buff != null) {
          buff.close();
        }
        if (buff3 != null) {
          buff3.close();
        }
        if (buff2 != null) {
          buff2.close();
        }
      }
      log.warn("DONE TEST FOR size = " + i + "\n");
    }
  }

  @Test
  void testFragmentationAndExhaustion() {
    final long poolSize = 15 * 1024L;
    PinnedMemoryPool.initialize(poolSize);
    assertEquals(poolSize, PinnedMemoryPool.getAvailableBytes());
    HostMemoryBuffer[] buffers = new HostMemoryBuffer[5];
    try {
      buffers[0] = PinnedMemoryPool.tryAllocate(1024);
      assertNotNull(buffers[0]);
      assertEquals(14*1024L, PinnedMemoryPool.getAvailableBytes());
      buffers[1] = PinnedMemoryPool.tryAllocate(2048);
      assertNotNull(buffers[1]);
      assertEquals(12*1024L, PinnedMemoryPool.getAvailableBytes());
      buffers[2] = PinnedMemoryPool.tryAllocate(4096);
      assertNotNull(buffers[2]);
      assertEquals(8*1024L, PinnedMemoryPool.getAvailableBytes());
      buffers[1].close();
      assertEquals(10*1024L, PinnedMemoryPool.getAvailableBytes());
      buffers[1] = null;
      buffers[1] = PinnedMemoryPool.tryAllocate(8192);
      assertNotNull(buffers[1]);
      assertEquals(2*1024L, PinnedMemoryPool.getAvailableBytes());
      buffers[3] = PinnedMemoryPool.tryAllocate(2048);
      assertNotNull(buffers[3]);
      assertEquals(0L, PinnedMemoryPool.getAvailableBytes());
      buffers[4] = PinnedMemoryPool.tryAllocate(64);
      assertNull(buffers[4]);
      buffers[0].close();
      assertEquals(1024L, PinnedMemoryPool.getAvailableBytes());
      buffers[0] = null;
      buffers[4] = PinnedMemoryPool.tryAllocate(64);
      assertNotNull(buffers[4]);
      assertEquals(1024L - 64, PinnedMemoryPool.getAvailableBytes());
    } finally {
      for (HostMemoryBuffer buffer : buffers) {
        if (buffer != null) {
          buffer.close();
        }
      }
    }
    assertEquals(poolSize, PinnedMemoryPool.getAvailableBytes());
  }

  @Test
  void testZeroSizedAllocation() {
    final long poolSize = 4 * 1024L;
    PinnedMemoryPool.initialize(poolSize);
    assertEquals(poolSize, PinnedMemoryPool.getAvailableBytes());
    try (HostMemoryBuffer buffer = PinnedMemoryPool.tryAllocate(0)) {
      assertNotNull(buffer);
      assertEquals(0, buffer.getLength());
      assertEquals(poolSize, PinnedMemoryPool.getAvailableBytes());
    }
    assertEquals(poolSize, PinnedMemoryPool.getAvailableBytes());
  }
}
