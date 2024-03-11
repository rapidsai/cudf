/*
 *
 *  Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

import java.nio.ByteBuffer;
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
    assertEquals(poolSize, PinnedMemoryPool.getTotalPoolSizeBytes());
    HostMemoryBuffer[] buffers = new HostMemoryBuffer[5];
    try {
      buffers[0] = PinnedMemoryPool.tryAllocate(1024);
      assertNotNull(buffers[0]);
      buffers[1] = PinnedMemoryPool.tryAllocate(2048);
      assertNotNull(buffers[1]);
      buffers[2] = PinnedMemoryPool.tryAllocate(4096);
      assertNotNull(buffers[2]);
      buffers[1].close();
      buffers[1] = null;
      buffers[1] = PinnedMemoryPool.tryAllocate(8192);
      assertNotNull(buffers[1]);
      buffers[3] = PinnedMemoryPool.tryAllocate(2048);
      assertNotNull(buffers[3]);
      buffers[4] = PinnedMemoryPool.tryAllocate(64);
      assertNull(buffers[4]);
      buffers[0].close();
      buffers[0] = null;
      buffers[4] = PinnedMemoryPool.tryAllocate(64);
      assertNotNull(buffers[4]);
    } finally {
      for (HostMemoryBuffer buffer : buffers) {
        if (buffer != null) {
          buffer.close();
        }
      }
    }
  }

  @Test
  void testTouchPinnedMemory() {
    final long poolSize = 15 * 1024L;
    PinnedMemoryPool.initialize(poolSize);
    int bufLength = 256;
    try(HostMemoryBuffer hmb = PinnedMemoryPool.allocate(bufLength);
        HostMemoryBuffer hmb2 = PinnedMemoryPool.allocate(bufLength)) {
      ByteBuffer bb = hmb.asByteBuffer(0, bufLength);
      for (int i = 0; i < bufLength; i++) {
        bb.put(i, (byte)i);
      }
      hmb2.copyFromHostBuffer(0, hmb, 0, bufLength);
      ByteBuffer bb2 = hmb2.asByteBuffer(0, bufLength);
      for (int i = 0; i < bufLength; i++) {
        assertEquals(bb.get(i), bb2.get(i));
      }
    }
  }

  @Test
  void testZeroSizedAllocation() {
    final long poolSize = 4 * 1024L;
    PinnedMemoryPool.initialize(poolSize);
    assertEquals(poolSize, PinnedMemoryPool.getTotalPoolSizeBytes());
    try (HostMemoryBuffer buffer = PinnedMemoryPool.tryAllocate(0)) {
      assertNotNull(buffer);
      assertEquals(0, buffer.getLength());
    }
  }

  // This test simulates cuIO using our fallback pinned pool wrapper
  // we should be able to either go to the pool, in this case 15KB in size
  // or we should be falling back to pinned cudaMallocHost/cudaFreeHost.
  @Test
  void testFallbackPinnedPool() {
    final long poolSize = 15 * 1024L;
    PinnedMemoryPool.initialize(poolSize);
    assertEquals(poolSize, PinnedMemoryPool.getTotalPoolSizeBytes());

    long ptr = Rmm.allocFromFallbackPinnedPool(1347);  // this doesn't fallback
    long ptr2 = Rmm.allocFromFallbackPinnedPool(15 * 1024L);  // this does
    Rmm.freeFromFallbackPinnedPool(ptr, 1347); // free from pool
    Rmm.freeFromFallbackPinnedPool(ptr2, 15*1024); // free from fallback

    ptr = Rmm.allocFromFallbackPinnedPool(15*1024L); // this doesn't fallback
    ptr2 = Rmm.allocFromFallbackPinnedPool(15*1024L); // this does
    Rmm.freeFromFallbackPinnedPool(ptr, 15*1024L); // free from pool
    Rmm.freeFromFallbackPinnedPool(ptr2, 15*1024L); // free from fallback
  }
}
