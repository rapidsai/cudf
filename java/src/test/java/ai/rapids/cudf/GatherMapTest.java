/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class GatherMapTest {
  private static final HostMemoryAllocator hostMemoryAllocator = DefaultHostMemoryAllocator.get();

  @Test
  void testInvalidBuffer() {
    try (DeviceMemoryBuffer buffer = DeviceMemoryBuffer.allocate(707)) {
      assertThrows(IllegalArgumentException.class, () -> new GatherMap(buffer));
    }
  }

  @Test
  void testRowCount() {
    try (GatherMap map = new GatherMap(DeviceMemoryBuffer.allocate(700))) {
      assertEquals(175, map.getRowCount());
    }
  }

  @Test
  void testClose() {
    DeviceMemoryBuffer mockBuffer = Mockito.mock(DeviceMemoryBuffer.class);
    GatherMap map = new GatherMap(mockBuffer);
    map.close();
    Mockito.verify(mockBuffer).close();
  }

  @Test
  void testReleaseBuffer() {
    DeviceMemoryBuffer mockBuffer = Mockito.mock(DeviceMemoryBuffer.class);
    GatherMap map = new GatherMap(mockBuffer);
    DeviceMemoryBuffer buffer = map.releaseBuffer();
    assertSame(mockBuffer, buffer);
    map.close();
    Mockito.verify(mockBuffer, Mockito.never()).close();
  }

  @Test
  void testInvalidColumnView() {
    try (GatherMap map = new GatherMap(DeviceMemoryBuffer.allocate(1024))) {
      assertThrows(IllegalArgumentException.class, () -> map.toColumnView(0, 257));
      assertThrows(IllegalArgumentException.class, () -> map.toColumnView(257, 0));
      assertThrows(IllegalArgumentException.class, () -> map.toColumnView(-4, 253));
      assertThrows(IllegalArgumentException.class, () -> map.toColumnView(4, -2));
    }
  }

  @Test
  void testToColumnView() {
    try (HostMemoryBuffer hostBuffer = hostMemoryAllocator.allocate(8 * 4)) {
      hostBuffer.setInts(0, new int[]{10, 11, 12, 13, 14, 15, 16, 17}, 0, 8);
      try (DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(8*4)) {
        devBuffer.copyFromHostBuffer(hostBuffer);
        devBuffer.incRefCount();
        try (GatherMap map = new GatherMap(devBuffer)) {
          ColumnView view = map.toColumnView(0, 8);
          assertEquals(DType.INT32, view.getType());
          assertEquals(0, view.getNullCount());
          assertEquals(8, view.getRowCount());
          try (HostMemoryBuffer viewHostBuffer = hostMemoryAllocator.allocate(8 * 4)) {
            viewHostBuffer.copyFromDeviceBuffer(view.getData());
            for (int i = 0; i < 8; i++) {
              assertEquals(i + 10, viewHostBuffer.getInt(4*i));
            }
          }
          view = map.toColumnView(3, 2);
          assertEquals(DType.INT32, view.getType());
          assertEquals(0, view.getNullCount());
          assertEquals(2, view.getRowCount());
          try (HostMemoryBuffer viewHostBuffer = hostMemoryAllocator.allocate(8)) {
            viewHostBuffer.copyFromDeviceBuffer(view.getData());
            assertEquals(13, viewHostBuffer.getInt(0));
            assertEquals(14, viewHostBuffer.getInt(4));
          }
        }
      }
    }
  }
}
