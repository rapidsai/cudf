/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

  private static final HostMemoryAllocator hostMemoryAllocator = DefaultHostMemoryAllocator.get();

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
    try (HostMemoryBuffer orig = hostMemoryAllocator.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer dest = hostMemoryAllocator.allocate(16)) {
      orig.setLong(0, 123456789);
      from.copyFromHostBuffer(orig);
      CuFile.writeDeviceBufferToFile(tempFile, 0, from);
      CuFile.readFileToDeviceBuffer(to, tempFile, 0);
      dest.copyFromDeviceBuffer(to);
      assertEquals(123456789, dest.getLong(0));
    }
  }

  private void verifyAppendToFile(File tempFile) {
    try (HostMemoryBuffer orig = hostMemoryAllocator.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer dest = hostMemoryAllocator.allocate(16)) {
      orig.setLong(0, 123456789);
      from.copyFromHostBuffer(orig);
      assertEquals(0, CuFile.appendDeviceBufferToFile(tempFile, from));

      orig.setLong(0, 987654321);
      from.copyFromHostBuffer(orig);
      assertEquals(16, CuFile.appendDeviceBufferToFile(tempFile, from));

      CuFile.readFileToDeviceBuffer(to, tempFile, 0);
      dest.copyFromDeviceBuffer(to);
      assertEquals(123456789, dest.getLong(0));

      CuFile.readFileToDeviceBuffer(to, tempFile, 16);
      dest.copyFromDeviceBuffer(to);
      assertEquals(987654321, dest.getLong(0));
    }
  }

  @Test
  public void testRegisteringUnalignedBufferThrowsException() {
    assumeTrue(CuFile.libraryLoaded());
    assertThrows(IllegalArgumentException.class, () -> {
      //noinspection EmptyTryBlock
      try (CuFileBuffer ignored = CuFileBuffer.allocate(4095, true)) {
      }
    });
  }

  @Test
  public void testReadWriteUnregisteredBuffer(@TempDir File tempDir) {
    assumeTrue(CuFile.libraryLoaded());
    File tempFile = new File(tempDir, "tempFile");
    verifyReadWrite(tempFile, 16, false);
  }

  @Test
  public void testReadWriteRegisteredBuffer(@TempDir File tempDir) {
    assumeTrue(CuFile.libraryLoaded());
    File tempFile = new File(tempDir, "tempFile");
    verifyReadWrite(tempFile, 4096, true);
  }

  private void verifyReadWrite(File tempFile, int length, boolean registerBuffer) {
    try (HostMemoryBuffer orig = hostMemoryAllocator.allocate(length);
         CuFileBuffer from = CuFileBuffer.allocate(length, registerBuffer);
         CuFileWriteHandle writer = new CuFileWriteHandle(tempFile.getAbsolutePath())) {
      orig.setLong(0, 123456789);
      from.copyFromHostBuffer(orig);
      writer.write(from, length, 0);

      orig.setLong(0, 987654321);
      from.copyFromHostBuffer(orig);
      assertEquals(length, writer.append(from, length));
    }
    try (CuFileBuffer to = CuFileBuffer.allocate(length, registerBuffer);
         CuFileReadHandle reader = new CuFileReadHandle(tempFile.getAbsolutePath());
         HostMemoryBuffer dest = hostMemoryAllocator.allocate(length)) {
      reader.read(to, 0);
      dest.copyFromDeviceBuffer(to);
      assertEquals(123456789, dest.getLong(0));

      reader.read(to, length);
      dest.copyFromDeviceBuffer(to);
      assertEquals(987654321, dest.getLong(0));
    }
  }
}
