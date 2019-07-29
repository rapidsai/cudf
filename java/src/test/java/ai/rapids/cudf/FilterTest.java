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

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class FilterTest {

  @Test
  void testMaskWithValidity() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    final int numRows = 5;
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.BOOL8, numRows)) {
      for (int i = 0; i < numRows; ++i) {
        builder.append((byte) 1);
        if (i % 2 != 0) {
          builder.setNullAt(i);
        }
      }
      try (ColumnVector mask = builder.build();
           ColumnVector input = ColumnVector.fromBoxedInts(1, null, 2, 3, null);
           ColumnVector filtered = input.filter(mask)) {
        filtered.ensureOnHost();
        assertEquals(input.getType(), filtered.getType());
        assertEquals(3, filtered.getRowCount());
        assertEquals(1, filtered.getInt(0));
        assertEquals(2, filtered.getInt(1));
        assertTrue(filtered.isNull(2));
      }
    }
  }

  @Test
  void testMaskDataOnly() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    byte[] maskVals = new byte[]{0, 1, 0, 1, 1};
    try (ColumnVector mask = ColumnVector.boolFromBytes(maskVals);
         ColumnVector input = ColumnVector.fromBoxedBytes((byte) 1, null, (byte) 2, (byte) 3, null);
         ColumnVector filtered = input.filter(mask)) {
      filtered.ensureOnHost();
      assertEquals(input.getType(), filtered.getType());
      assertEquals(3, filtered.getRowCount());
      assertTrue(filtered.isNull(0));
      assertEquals(3, filtered.getByte(1));
      assertTrue(filtered.isNull(2));
    }
  }

  @Test
  void testAllFilteredFromData() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    Boolean[] maskVals = new Boolean[5];
    Arrays.fill(maskVals, false);
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(maskVals);
         ColumnVector input = ColumnVector.fromBoxedInts(1, null, 2, 3, null);
         ColumnVector filtered = input.filter(mask)) {
      assertEquals(input.getType(), filtered.getType());
      assertEquals(0, filtered.getRowCount());
    }
  }

  @Test
  void testAllFilteredFromValidity() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    final int numRows = 5;
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.BOOL8, numRows)) {
      for (int i = 0; i < numRows; ++i) {
        builder.append((byte) 1);
        builder.setNullAt(i);
      }
      try (ColumnVector mask = builder.build();
           ColumnVector input = ColumnVector.fromBoxedInts(1, null, 2, 3, null);
           ColumnVector filtered = input.filter(mask)) {
        assertEquals(input.getType(), filtered.getType());
        assertEquals(0, filtered.getRowCount());
      }
    }
  }

  @Test
  void testMismatchedSizes() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    Boolean[] maskVals = new Boolean[3];
    Arrays.fill(maskVals, true);
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(maskVals);
         ColumnVector input = ColumnVector.fromBoxedInts(1, null, 2, 3, null)) {
      assertThrows(AssertionError.class, () -> input.filter(mask).close());
    }
  }
}
