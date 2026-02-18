/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostMemoryBuffer;
import com.google.common.io.Closer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;


class TableTestUtils {
  public static byte[] arrayFrom(File f) throws IOException {
    long len = f.length();
    if (len > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("Sorry cannot read " + f +
              " into an array it does not fit");
    }
    int remaining = (int)len;
    byte[] ret = new byte[remaining];
    try (FileInputStream fin = new FileInputStream(f)) {
      int at = 0;
      while (remaining > 0) {
        int amount = fin.read(ret, at, remaining);
        at += amount;
        remaining -= amount;
      }
    }
    return ret;
  }

  public static byte[][] sliceBytes(byte[] data, int slices) {
    slices = Math.min(data.length, slices);
    // We are not going to worry about making it super even here.
    // The last one gets the extras.
    int bytesPerSlice = data.length / slices;
    byte[][] ret = new byte[slices][];
    int startingAt = 0;
    for (int i = 0; i < (slices - 1); i++) {
      ret[i] = new byte[bytesPerSlice];
      System.arraycopy(data, startingAt, ret[i], 0, bytesPerSlice);
      startingAt += bytesPerSlice;
    }
    // Now for the last one
    ret[slices - 1] = new byte[data.length - startingAt];
    System.arraycopy(data, startingAt, ret[slices - 1], 0, data.length - startingAt);
    return ret;
  }

  public static class HostMemoryBufferArray implements AutoCloseable {
    public final HostMemoryBuffer[] buffers;

    public HostMemoryBufferArray(HostMemoryBuffer[] buffers) {
      this.buffers = buffers;
    }

    @Override
    public void close() throws IOException {
      try (Closer closer = Closer.create()) {
        Arrays.stream(buffers).forEach(b -> closer.register(() -> b.close()));
      }
    }
  }

  public static HostMemoryBufferArray buffersFrom(byte[][] data) {
    HostMemoryBuffer[] buffers = new HostMemoryBuffer[data.length];
    for (int i = 0; i < data.length; i++) {
      byte[] subData = data[i];
      buffers[i] = HostMemoryBuffer.allocate(subData.length);
      buffers[i].setBytes(0, subData, 0, subData.length);
    }
    return new HostMemoryBufferArray(buffers);
  }
}
