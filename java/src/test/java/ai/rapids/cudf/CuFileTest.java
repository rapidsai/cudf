package ai.rapids.cudf;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class CuFileTest extends CudfTestBase {
  @TempDir File tempDir;

  @AfterEach
  void tearDown() {
    if (PinnedMemoryPool.isInitialized()) {
      PinnedMemoryPool.shutdown();
    }
  }

  @Test
  public void testCopyToFile() {
    assumeTrue(CuFile.libraryLoaded());
    File tempFile = new File(tempDir, "tempFile");
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

  @Test
  public void testAppendToFile() {
    assumeTrue(CuFile.libraryLoaded());
    File tempFile = new File(tempDir, "tempFile");
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
