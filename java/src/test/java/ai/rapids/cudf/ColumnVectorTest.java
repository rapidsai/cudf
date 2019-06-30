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
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.mock;

public class ColumnVectorTest {

  @Test
  void testCudfColumnSize() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    DeviceMemoryBuffer mockDataBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
    DeviceMemoryBuffer mockValidBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);

    try (ColumnVector v0 = new ColumnVector(DType.INT32, TimeUnit.NONE, 0, mockDataBuffer,
        mockValidBuffer)) {
      v0.getNativeCudfColumnAddress();
    }

    try (ColumnVector v1 = new ColumnVector(DType.INT32, TimeUnit.NONE, Long.MAX_VALUE,
        mockDataBuffer, mockValidBuffer)) {
      assertThrows(AssertionError.class, () -> v1.getNativeCudfColumnAddress());
    }
  }

  @Test
  void testCudfColumnFromHostVector() {
    HostMemoryBuffer mockDataBuffer = mock(HostMemoryBuffer.class);
    try (ColumnVector v = new ColumnVector(DType.INT32, TimeUnit.NONE, 10, 0, mockDataBuffer,
        null)) {
      assertThrows(IllegalStateException.class, () -> v.getNativeCudfColumnAddress());
    }
  }

  @Test
  void testGetNativeAddressFromHostVector() {
    HostMemoryBuffer mockDataBuffer = mock(HostMemoryBuffer.class);
    try (ColumnVector v = new ColumnVector(DType.INT32, TimeUnit.NONE, 10, 0, mockDataBuffer,
        null)) {
      assertThrows(IllegalStateException.class, () -> v.getNativeCudfColumnAddress());
    }
  }

  @Test
  void testRefCount() {
    DeviceMemoryBuffer mockDataBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
    DeviceMemoryBuffer mockValidBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);

    assertThrows(IllegalStateException.class, () -> {
      try (ColumnVector v2 = new ColumnVector(DType.INT32, TimeUnit.NONE, Long.MAX_VALUE,
          mockDataBuffer, mockValidBuffer)) {
        v2.close();
      }
    });
  }

  @Test
  void testRefCountLeak() throws InterruptedException {
    assumeTrue(Boolean.getBoolean("ai.rapids.cudf.flaky-tests-enabled"));
    long expectedLeakCount = ColumnVectorCleaner.leakCount.get() + 1;
    DeviceMemoryBuffer mockDataBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
    DeviceMemoryBuffer mockValidBuffer = mock(DeviceMemoryBuffer.class, Mockito.RETURNS_DEEP_STUBS);
    new ColumnVector(DType.INT32, TimeUnit.NONE, Long.MAX_VALUE, mockDataBuffer, mockValidBuffer);
    long maxTime = System.currentTimeMillis() + 10_000;
    long leakNow;
    do {
      System.gc();
      Thread.sleep(50);
      leakNow = ColumnVectorCleaner.leakCount.get();
    } while (leakNow != expectedLeakCount && System.currentTimeMillis() < maxTime);
    assertEquals(expectedLeakCount, ColumnVectorCleaner.leakCount.get());
  }

  @Test
  void testConcatTypeError() {
    try (ColumnVector v0 = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromFloats(5.0f, 6.0f)) {
      assertThrows(CudfException.class, () -> ColumnVector.concatenate(v0, v1));
    }
  }

  @Test
  void testConcatNoNulls() {
    try (ColumnVector v0 = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromInts(5, 6, 7);
         ColumnVector v2 = ColumnVector.fromInts(8, 9);
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2)) {
      v.ensureOnHost();
      assertEquals(9, v.getRowCount());
      assertFalse(v.hasNulls());
      assertFalse(v.hasValidityVector());
      for (int i = 0; i < 9; ++i) {
        assertEquals(i + 1, v.getInt(i), "at index " + i);
      }
    }
  }

  @Test
  void testConcatWithNulls() {
    try (ColumnVector v0 = ColumnVector.fromDoubles(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromDoubles(5, 6, 7);
         ColumnVector v2 = ColumnVector.fromBoxedDoubles(null, 9.0);
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2)) {
      v.ensureOnHost();
      assertEquals(9, v.getRowCount());
      assertTrue(v.hasNulls());
      assertTrue(v.hasValidityVector());
      for (int i = 0; i < 9; ++i) {
        if (i != 7) {
          assertEquals(i + 1, v.getDouble(i), "at index " + i);
        } else {
          assertTrue(v.isNull(i), "at index " + i);
        }
      }
    }
  }
}
