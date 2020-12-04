package ai.rapids.cudf;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
    File tempFile = new File(tempDir, "tempFile");
    try (HostMemoryBuffer orig = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer from = DeviceMemoryBuffer.allocate(16);
         DeviceMemoryBuffer to = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer dest = HostMemoryBuffer.allocate(16);) {
      orig.setLong(0, 123456789);
      from.copyFromHostBuffer(orig);
      CuFile.copyDeviceBufferToFile(tempFile, from);
      CuFile.copyFileToDeviceBuffer(to, tempFile);
      dest.copyFromDeviceBuffer(to);
      assertEquals(123456789, dest.getLong(0));
    }
  }
}
