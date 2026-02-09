/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

public class MemoryBufferTest extends CudfTestBase {
  private static final HostMemoryAllocator hostMemoryAllocator = DefaultHostMemoryAllocator.get();

  private static final byte[] BYTES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  private static final byte[] EXPECTED = {0, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  @Test
  public void testAddressOutOfBoundsExceptionWhenCopying() {
    try (HostMemoryBuffer from = hostMemoryAllocator.allocate(16);
         HostMemoryBuffer to = hostMemoryAllocator.allocate(16)) {
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
    try (HostMemoryBuffer from = hostMemoryAllocator.allocate(16);
         HostMemoryBuffer to = hostMemoryAllocator.allocate(16)) {
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
    try (HostMemoryBuffer in = hostMemoryAllocator.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer out = hostMemoryAllocator.allocate(16)) {
      in.setBytes(0, BYTES, 0, 16);
      from.copyFromHostBuffer(in);
      to.copyFromMemoryBuffer(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      to.copyFromMemoryBuffer(1, from, 2, 3, Cuda.DEFAULT_STREAM);
      out.copyFromDeviceBuffer(to);
      verifyOutput(out);
    }
  }

  @Test
  public void testCopyingFromDeviceToDeviceAsync() {
    try (HostMemoryBuffer in = hostMemoryAllocator.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer out = hostMemoryAllocator.allocate(16)) {
      in.setBytes(0, BYTES, 0, 16);
      from.copyFromHostBuffer(in);
      to.copyFromMemoryBufferAsync(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      to.copyFromMemoryBufferAsync(1, from, 2, 3, Cuda.DEFAULT_STREAM);
      out.copyFromDeviceBufferAsync(to, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();
      verifyOutput(out);
    }
  }

  @Test
  public void testCopyingFromHostToHost() {
    try (HostMemoryBuffer from = hostMemoryAllocator.allocate(16);
         HostMemoryBuffer to = hostMemoryAllocator.allocate(16)) {
      from.setBytes(0, BYTES, 0, 16);
      to.setBytes(0, BYTES, 0, 16);
      to.copyFromMemoryBuffer(1, from, 2, 3, Cuda.DEFAULT_STREAM);
      verifyOutput(to);
    }
  }

  @Test
  public void testCopyingFromHostToHostAsync() {
    try (HostMemoryBuffer from = hostMemoryAllocator.allocate(16);
         HostMemoryBuffer to = hostMemoryAllocator.allocate(16)) {
      from.setBytes(0, BYTES, 0, 16);
      to.setBytes(0, BYTES, 0, 16);
      to.copyFromMemoryBufferAsync(1, from, 2, 3, Cuda.DEFAULT_STREAM);
      verifyOutput(to);
    }
  }

  @Test
  public void testCopyingFromHostToDevice() {
    try (HostMemoryBuffer from = hostMemoryAllocator.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer out = hostMemoryAllocator.allocate(16)) {
      from.setBytes(0, BYTES, 0, 16);
      to.copyFromMemoryBuffer(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      to.copyFromMemoryBufferAsync(1, from, 2, 3, Cuda.DEFAULT_STREAM);
      out.copyFromDeviceBuffer(to);
      verifyOutput(out);
    }
  }

  @Test
  public void testCopyingFromHostToDeviceAsync() {
    try (HostMemoryBuffer from = hostMemoryAllocator.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer out = hostMemoryAllocator.allocate(16)) {
      from.setBytes(0, BYTES, 0, 16);
      to.copyFromMemoryBufferAsync(0, from, 0, 16, Cuda.DEFAULT_STREAM);
      to.copyFromMemoryBufferAsync(1, from, 2, 3, Cuda.DEFAULT_STREAM);
      out.copyFromDeviceBufferAsync(to, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();
      verifyOutput(out);
    }
  }

  @Test
  public void testCopyingFromDeviceToHost() {
    try (HostMemoryBuffer in = hostMemoryAllocator.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer to = hostMemoryAllocator.allocate(16)) {
      in.setBytes(0, BYTES, 0, 16);
      from.copyFromHostBuffer(in);
      to.setBytes(0, BYTES, 0, 16);
      to.copyFromMemoryBuffer(1, from, 2, 3, Cuda.DEFAULT_STREAM);
      verifyOutput(to);
    }
  }

  @Test
  public void testCopyingFromDeviceToHostAsync() {
    try (HostMemoryBuffer in = hostMemoryAllocator.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer to = hostMemoryAllocator.allocate(16)) {
      in.setBytes(0, BYTES, 0, 16);
      from.copyFromHostBuffer(in);
      to.setBytes(0, BYTES, 0, 16);
      to.copyFromMemoryBufferAsync(1, from, 2, 3, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();
      verifyOutput(to);
    }
  }

  private void verifyOutput(HostMemoryBuffer out) {
    byte[] bytes = new byte[16];
    out.getBytes(bytes, 0, 0, 16);
    assertArrayEquals(EXPECTED, bytes);
  }

  @Test
  public void testEventHandlerIsCalledForEachClose() {
    final AtomicInteger onClosedWasCalled = new AtomicInteger(0);
    try (DeviceMemoryBuffer b = DeviceMemoryBuffer.allocate(256)) {
      b.setEventHandler(refCount -> onClosedWasCalled.incrementAndGet());
    }
    assertEquals(1, onClosedWasCalled.get());
    onClosedWasCalled.set(0);

    try (DeviceMemoryBuffer b = DeviceMemoryBuffer.allocate(256)) {
      b.setEventHandler(refCount -> onClosedWasCalled.incrementAndGet());
      DeviceMemoryBuffer sliced = b.slice(0, b.getLength());
      sliced.close();
    }
    assertEquals(2, onClosedWasCalled.get());
  }

  @Test
  public void testEventHandlerIsNotCalledIfNotSet() {
    final AtomicInteger onClosedWasCalled = new AtomicInteger(0);
    try (DeviceMemoryBuffer b = DeviceMemoryBuffer.allocate(256)) {
      assertNull(b.getEventHandler());
    }
    assertEquals(0, onClosedWasCalled.get());
    try (DeviceMemoryBuffer b = DeviceMemoryBuffer.allocate(256)) {
      b.setEventHandler(refCount -> onClosedWasCalled.incrementAndGet());
      b.setEventHandler(null);
    }
    assertEquals(0, onClosedWasCalled.get());
  }

  @Test
  public void testEventHandlerReturnsPreviousHandlerOnReset() {
    try (DeviceMemoryBuffer b = DeviceMemoryBuffer.allocate(256)) {
      MemoryBuffer.EventHandler handler = refCount -> {};
      MemoryBuffer.EventHandler handler2 = refCount -> {};

      assertNull(b.setEventHandler(handler));
      assertEquals(handler, b.setEventHandler(null));

      assertNull(b.setEventHandler(handler2));
      assertEquals(handler2, b.setEventHandler(handler));
    }
  }
}
