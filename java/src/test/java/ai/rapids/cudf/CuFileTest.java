/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class CuFileTest extends CudfTestBase {
  @AfterEach
  void tearDown() {
    if (PinnedMemoryPool.isInitialized()) {
      PinnedMemoryPool.shutdown();
    }
  }

  @Test
  public void testCopyToFile(@TempDir File tempDir) {
    assumeTrue(CuFile.libraryLoaded());
    File tempFile = new File(tempDir, "tempFile");
    assertFalse(tempFile.exists());
    verifyCopyToFile(tempFile);
  }

  @Test
  public void testCopyToExistingFile(@TempDir File tempDir) throws IOException {
    assumeTrue(CuFile.libraryLoaded());
    File tempFile = new File(tempDir, "tempFile");
    assertTrue(tempFile.createNewFile());
    verifyCopyToFile(tempFile);
  }

  @Test
  public void testAppendToFile(@TempDir File tempDir) {
    assumeTrue(CuFile.libraryLoaded());
    File tempFile = new File(tempDir, "tempFile");
    assertFalse(tempFile.exists());
    verifyAppendToFile(tempFile);
  }

  @Test
  public void testAppendToExistingFile(@TempDir File tempDir) throws IOException {
    assumeTrue(CuFile.libraryLoaded());
    File tempFile = new File(tempDir, "tempFile");
    assertTrue(tempFile.createNewFile());
    verifyAppendToFile(tempFile);
  }

  private void verifyCopyToFile(File tempFile) {
    try (HostMemoryBuffer orig = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer dest = HostMemoryBuffer.allocate(16);) {
      orig.setLong(0, 123456789);
      from.copyFromHostBuffer(orig);
      CuFile.writeDeviceBufferToFile(tempFile, 0, from);
      CuFile.readFileToDeviceBuffer(to, tempFile, 0);
      dest.copyFromDeviceBuffer(to);
      assertEquals(123456789, dest.getLong(0));
    }
  }

  private void verifyAppendToFile(File tempFile) {
    try (HostMemoryBuffer orig = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer dest = HostMemoryBuffer.allocate(16);) {
      orig.setLong(0, 123456789);
      from.copyFromHostBuffer(orig);
      CuFile.appendDeviceBufferToFile(tempFile, from);

      orig.setLong(0, 987654321);
      from.copyFromHostBuffer(orig);
      CuFile.appendDeviceBufferToFile(tempFile, from);

      CuFile.readFileToDeviceBuffer(to, tempFile, 0);
      dest.copyFromDeviceBuffer(to);
      assertEquals(123456789, dest.getLong(0));

      CuFile.readFileToDeviceBuffer(to, tempFile, 16);
      dest.copyFromDeviceBuffer(to);
      assertEquals(987654321, dest.getLong(0));
    }
  }
}
