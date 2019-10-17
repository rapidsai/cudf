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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class HostMemoryBufferTest extends CudfTestBase {
  @Test
  void testRefCountLeak() throws InterruptedException {
    assumeTrue(Boolean.getBoolean("ai.rapids.cudf.flaky-tests-enabled"));
    long expectedLeakCount = MemoryCleaner.leakCount.get() + 1;
    HostMemoryBuffer.allocate(1);
    long maxTime = System.currentTimeMillis() + 10_000;
    long leakNow;
    do {
      System.gc();
      Thread.sleep(50);
      leakNow = MemoryCleaner.leakCount.get();
    } while (leakNow != expectedLeakCount && System.currentTimeMillis() < maxTime);
    assertEquals(expectedLeakCount, MemoryCleaner.leakCount.get());
  }

  @Test
  void testDoubleFree() {
    HostMemoryBuffer buffer = HostMemoryBuffer.allocate(1);
    buffer.close();
    assertThrows(IllegalStateException.class, () -> buffer.close() );
  }

  @Test
  public void testGetInt() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      long offset = 1;
      hostMemoryBuffer.setInt(offset * DType.INT32.sizeInBytes, 2);
      assertEquals(2, hostMemoryBuffer.getInt(offset * DType.INT32.sizeInBytes));
    }
  }

  @Test
  public void testGetByte() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      long offset = 1;
      hostMemoryBuffer.setByte(offset * DType.INT8.sizeInBytes, (byte) 2);
      assertEquals((byte) 2, hostMemoryBuffer.getByte(offset * DType.INT8.sizeInBytes));
    }
  }

  @Test
  public void testGetLong() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      long offset = 1;
      hostMemoryBuffer.setLong(offset * DType.INT64.sizeInBytes, 3);
      assertEquals(3, hostMemoryBuffer.getLong(offset * DType.INT64.sizeInBytes));
    }
  }

  @Test
  public void testGetLength() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      long length = hostMemoryBuffer.getLength();
      assertEquals(16, length);
    }
  }

  @Test
  public void testCopyFromDeviceBuffer() {
    try (HostMemoryBuffer init = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer tmp = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer to = HostMemoryBuffer.allocate(16)) {
      init.setLong(0, 123456789);
      tmp.copyFromHostBuffer(init);
      to.copyFromDeviceBuffer(tmp);
      assertEquals(123456789, to.getLong(0));
    }
  }
}
