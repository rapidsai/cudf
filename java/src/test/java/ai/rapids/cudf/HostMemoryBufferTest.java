/*
 *
 *  Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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
  void asByteBuffer() {
    final long size = 1024;
    try (HostMemoryBuffer buff = HostMemoryBuffer.allocate(size)) {
      ByteBuffer dbuff = buff.asByteBuffer();
      assertEquals(size, dbuff.capacity());
      assertEquals(ByteOrder.nativeOrder(), dbuff.order());
      dbuff.putInt(101);
      dbuff.putDouble(101.1);
      assertEquals(101, buff.getInt(0));
      assertEquals(101.1, buff.getDouble(4));
    }
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

  @Test
  public void testFilemap() throws Exception {
    Random random = new Random(12345L);
    final int pageSize = UnsafeMemoryAccessor.pageSize();
    final int bufferSize = pageSize * 5;
    byte[] testbuf = new byte[bufferSize];
    random.nextBytes(testbuf);
    Path tempFile = Files.createTempFile("mmaptest", ".data");
    try {
      Files.write(tempFile, testbuf);

      // verify we can map the whole file
      try (HostMemoryBuffer hmb = HostMemoryBuffer.mapFile(tempFile.toFile(),
          FileChannel.MapMode.READ_ONLY, 0,bufferSize)) {
        assertEquals(bufferSize, hmb.length);
        byte[] bytes = new byte[(int) hmb.length];
        hmb.getBytes(bytes, 0, 0, hmb.length);
        assertArrayEquals(testbuf, bytes);
      }

      // verify we can map at offsets that aren't a page boundary
      int mapOffset = pageSize + 1;
      int mapLength = pageSize * 2 + 7;
      try (HostMemoryBuffer hmb = HostMemoryBuffer.mapFile(tempFile.toFile(),
          FileChannel.MapMode.READ_ONLY, mapOffset, mapLength)) {
        assertEquals(mapLength, hmb.length);
        byte[] expected = Arrays.copyOfRange(testbuf, mapOffset, mapOffset + mapLength);
        byte[] bytes = new byte[(int) hmb.length];
        hmb.getBytes(bytes, 0, 0, hmb.length);
        assertArrayEquals(expected, bytes);
      }

      // verify we can modify the file via a writable mapping
      mapOffset = pageSize * 3 + 123;
      mapLength = bufferSize - mapOffset - 456;
      byte[] newData = new byte[mapLength];
      random.nextBytes(newData);
      try (HostMemoryBuffer hmb = HostMemoryBuffer.mapFile(tempFile.toFile(),
          FileChannel.MapMode.READ_WRITE, mapOffset, mapLength)) {
        hmb.setBytes(0, newData, 0, newData.length);
      }
      byte[] data = Files.readAllBytes(tempFile);
      System.arraycopy(newData, 0, testbuf, mapOffset, mapLength);
      assertArrayEquals(testbuf, data);
    } finally {
      Files.delete(tempFile);
    }
  }


  @Test
  public void testCopyWithStream() {
    Random random = new Random(12345L);
    byte[] data = new byte[4096];
    byte[] result = new byte[data.length];
    random.nextBytes(data);
    try (Cuda.Stream stream1 = new Cuda.Stream(true);
         Cuda.Stream stream2 = new Cuda.Stream(true);
         HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(data.length);
         DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(data.length);
         HostMemoryBuffer hostBuffer2 = HostMemoryBuffer.allocate(data.length)) {
      hostBuffer.setBytes(0, data, 0, data.length);
      devBuffer.copyFromHostBuffer(hostBuffer, stream1);
      hostBuffer2.copyFromDeviceBuffer(devBuffer, stream2);
      hostBuffer2.getBytes(result, 0, 0, result.length);
      assertArrayEquals(data, result);
    }
  }
}
