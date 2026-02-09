/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RmmMemoryAccessorTest extends CudfTestBase {
  @Test
  public void log() throws IOException {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
    File f = File.createTempFile("ALL_LOG",".csv");
    f.deleteOnExit();
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, Rmm.logTo(f), 1024*1024*1024);
    try (DeviceMemoryBuffer address = Rmm.alloc(10, Cuda.DEFAULT_STREAM)) {
      assertNotEquals(0, address);
    }
    Rmm.shutdown();
    StringBuilder log = new StringBuilder();
    try (Stream<String> stream = Files.lines(f.toPath(), StandardCharsets.UTF_8))
    {
        stream.forEach(s -> log.append(s).append("\n"));
    }
    System.err.println(log);
    assertNotNull(log.toString());
    assertTrue(0 < log.length());
  }


  @Test
  public void init() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
    assertFalse(Rmm.isInitialized());
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, Rmm.logToStderr(), -1);
    assertTrue(Rmm.isInitialized());
    Rmm.shutdown();
    assertFalse(Rmm.isInitialized());
  }

  @Test
  public void shutdown() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
    Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 2048);
    try (DeviceMemoryBuffer buffer = DeviceMemoryBuffer.allocate(1024)) {
      assertThrows(RmmException.class, () -> Rmm.shutdown(500, 2000, TimeUnit.MILLISECONDS));
    }
    Rmm.shutdown();
  }

  @Test
  public void allocate() {
    try (DeviceMemoryBuffer address = Rmm.alloc(10, Cuda.DEFAULT_STREAM)) {
      assertNotEquals(0, address.address);
    }
  }

  @Test
  public void doubleInitFails() {
    if (!Rmm.isInitialized()) {
      Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, Rmm.logToStderr(), 0);
    }
    assertThrows(IllegalStateException.class,
        () -> Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 1024 * 1024));
  }
}
