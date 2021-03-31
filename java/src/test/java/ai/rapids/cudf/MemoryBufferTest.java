/*
 *
 *  Copyright (c) 2021, NVIDIA CORPORATION.
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

import static org.junit.jupiter.api.Assertions.*;

public class MemoryBufferTest extends CudfTestBase {
  @AfterEach
  void teardown() {
    if (PinnedMemoryPool.isInitialized()) {
      PinnedMemoryPool.shutdown();
    }
  }

  @Test
  public void testAddressOutOfBoundsExceptionWhenCopying() {
    try (HostMemoryBuffer from = HostMemoryBuffer.allocate(16);
         HostMemoryBuffer to = HostMemoryBuffer.allocate(16)) {
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBuffer(-1, from, 0, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBuffer(16, from, 0, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBuffer(0, from, -1, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBuffer(0, from, 16, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBuffer(0, from, 0, -1, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBuffer(0, from, 0, 17, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBuffer(1, from, 0, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBuffer(0, from, 1, 16, Cuda.DEFAULT_STREAM));
    }
  }

  @Test
  public void testAddressOutOfBoundsExceptionWhenCopyingAsync() {
    try (HostMemoryBuffer from = HostMemoryBuffer.allocate(16);
         HostMemoryBuffer to = HostMemoryBuffer.allocate(16)) {
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBufferAsync(-1, from, 0, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBufferAsync(16, from, 0, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBufferAsync(0, from, -1, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBufferAsync(0, from, 16, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBufferAsync(0, from, 0, -1, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBufferAsync(0, from, 0, 17, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBufferAsync(1, from, 0, 16, Cuda.DEFAULT_STREAM));
      assertThrows(AssertionError.class, () -> to.copyFromMemoryBufferAsync(0, from, 1, 16, Cuda.DEFAULT_STREAM));
    }
  }

  @Test
  public void testCopyingFromDeviceToDevice() {
    try (HostMemoryBuffer in = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer out = HostMemoryBuffer.allocate(16)) {
      in.setLong(0, 123456789);
      from.copyFromHostBuffer(in);
      to.copyFromMemoryBuffer(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      out.copyFromDeviceBuffer(to);
      assertEquals(123456789, out.getLong(0));
    }
  }

  @Test
  public void testCopyingFromDeviceToDeviceAsync() {
    try (HostMemoryBuffer in = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer out = HostMemoryBuffer.allocate(16)) {
      in.setLong(0, 123456789);
      from.copyFromHostBuffer(in);
      to.copyFromMemoryBufferAsync(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      out.copyFromDeviceBufferAsync(to, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();
      assertEquals(123456789, out.getLong(0));
    }
  }

  @Test
  public void testCopyingFromHostToHost() {
    try (HostMemoryBuffer from = HostMemoryBuffer.allocate(16);
         HostMemoryBuffer to = HostMemoryBuffer.allocate(16)) {
      from.setLong(0, 123456789);
      to.copyFromMemoryBuffer(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      assertEquals(123456789, to.getLong(0));
    }
  }

  @Test
  public void testCopyingFromHostToHostAsync() {
    try (HostMemoryBuffer from = HostMemoryBuffer.allocate(16);
         HostMemoryBuffer to = HostMemoryBuffer.allocate(16)) {
      from.setLong(0, 123456789);
      to.copyFromMemoryBufferAsync(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      assertEquals(123456789, to.getLong(0));
    }
  }

  @Test
  public void testCopyingFromHostToDevice() {
    try (HostMemoryBuffer from = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer out = HostMemoryBuffer.allocate(16)) {
      from.setLong(0, 123456789);
      to.copyFromMemoryBuffer(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      out.copyFromDeviceBuffer(to);
      assertEquals(123456789, out.getLong(0));
    }
  }

  @Test
  public void testCopyingFromHostToDeviceAsync() {
    try (HostMemoryBuffer from = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer out = HostMemoryBuffer.allocate(16)) {
      from.setLong(0, 123456789);
      to.copyFromMemoryBufferAsync(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      out.copyFromDeviceBufferAsync(to, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();
      assertEquals(123456789, out.getLong(0));
    }
  }

  @Test
  public void testCopyingFromDeviceToHost() {
    try (HostMemoryBuffer in = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer to = HostMemoryBuffer.allocate(16)) {
      in.setLong(0, 123456789);
      from.copyFromHostBuffer(in);
      to.copyFromMemoryBuffer(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      assertEquals(123456789, to.getLong(0));
    }
  }

  @Test
  public void testCopyingFromDeviceToHostAsync() {
    try (HostMemoryBuffer in = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer to = HostMemoryBuffer.allocate(16)) {
      in.setLong(0, 123456789);
      from.copyFromHostBuffer(in);
      to.copyFromMemoryBufferAsync(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();
      assertEquals(123456789, to.getLong(0));
    }
  }
}
