/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.examples;

import ai.rapids.cudf.ByteRange;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;

/**
 * Shared helpers for the hybrid-scan Java examples. Mirrors the lightweight pieces of
 * {@code cpp/examples/hybrid_scan_io/io_utils.hpp}.
 */
public final class Util {
  private Util() {}

  /** Magic + length suffix at the end of every Parquet file. */
  private static final int FOOTER_TAIL_BYTES = 8;

  /**
   * Copy the entire file contents into a {@link HostMemoryBuffer}.
   */
  public static HostMemoryBuffer readFileToHostBuffer(File file) throws IOException {
    byte[] bytes = Files.readAllBytes(file.toPath());
    HostMemoryBuffer buf = HostMemoryBuffer.allocate(bytes.length);
    buf.setBytes(0, bytes, 0, bytes.length);
    return buf;
  }

  /**
   * Slice the Parquet footer out of a host buffer that holds the full file.
   */
  public static HostMemoryBuffer extractFooter(HostMemoryBuffer file) {
    long fileLen = file.getLength();
    int footerLen = file.getInt(fileLen - FOOTER_TAIL_BYTES);
    long footerStart = fileLen - FOOTER_TAIL_BYTES - footerLen;
    return file.slice(footerStart, footerLen);
  }

  /**
   * Reads the 4-byte little-endian footer length field from the tail of an open Parquet file.
   * Leaves the file position undefined after returning.
   */
  private static int getFooterLength(RandomAccessFile raf) throws IOException {
    raf.seek(raf.length() - FOOTER_TAIL_BYTES);
    byte[] tail = new byte[4];
    raf.readFully(tail);
    return ByteBuffer.wrap(tail).order(ByteOrder.LITTLE_ENDIAN).getInt();
  }

  /**
   * Reads only the Parquet footer bytes from {@code file} without loading the rest of the
   * file into memory.
   */
  public static HostMemoryBuffer readFooterOnly(File file) throws IOException {
    try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
      int footerLen = getFooterLength(raf);
      byte[] bytes = new byte[footerLen];
      raf.seek(raf.length() - FOOTER_TAIL_BYTES - footerLen);
      raf.readFully(bytes);   // guaranteed complete read — no while loop needed
      HostMemoryBuffer footer = HostMemoryBuffer.allocate(footerLen);
      try {
        footer.setBytes(0, bytes, 0, footerLen);
        return footer;
      } catch (RuntimeException e) {
        footer.close();
        throw e;
      }
    }
  }

  /**
   * Reads the bytes described by {@code range} directly from {@code file} into a
   * new {@link HostMemoryBuffer}. Uses a single {@link RandomAccessFile} seek.
   * Caller owns the returned buffer and must close it.
   */
  public static HostMemoryBuffer readByteRange(File file, ByteRange range) throws IOException {
    try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
      byte[] bytes = new byte[(int) range.size()];
      raf.seek(range.offset());
      raf.readFully(bytes);
      HostMemoryBuffer buf = HostMemoryBuffer.allocate(range.size());
      try {
        buf.setBytes(0, bytes, 0, bytes.length);
        return buf;
      } catch (RuntimeException e) {
        buf.close();
        throw e;
      }
    }
  }

  /**
   * Copy each {@link ByteRange} from a host buffer into its own {@link DeviceMemoryBuffer}.
   * Caller owns the returned buffers and must close them.
   */
  public static DeviceMemoryBuffer[] copyRangesToDevice(HostMemoryBuffer file,
                                                        ByteRange[] ranges) {
    DeviceMemoryBuffer[] out = new DeviceMemoryBuffer[ranges.length];
    try {
      for (int i = 0; i < ranges.length; i++) {
        ByteRange r = ranges[i];
        DeviceMemoryBuffer dev = DeviceMemoryBuffer.allocate(r.size());
        try (HostMemoryBuffer slice = file.slice(r.offset(), r.size())) {
          dev.copyFromHostBuffer(slice);
        }
        out[i] = dev;
      }
      return out;
    } catch (Throwable t) {
      closeAll(out);
      throw t;
    }
  }

  /**
   * Reads each {@link ByteRange} from {@code file} directly (one seek per range using a
   * single open {@link RandomAccessFile}) and copies it into its own
   * {@link DeviceMemoryBuffer}. Caller owns the returned buffers and must close them.
   */
  public static DeviceMemoryBuffer[] copyRangesToDevice(File file, ByteRange[] ranges)
      throws IOException {
    DeviceMemoryBuffer[] out = new DeviceMemoryBuffer[ranges.length];
    try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
      for (int i = 0; i < ranges.length; i++) {
        ByteRange r = ranges[i];
        byte[] bytes = new byte[(int) r.size()];
        raf.seek(r.offset());
        raf.readFully(bytes);
        DeviceMemoryBuffer dev = DeviceMemoryBuffer.allocate(r.size());
        try (HostMemoryBuffer host = HostMemoryBuffer.allocate(r.size())) {
          host.setBytes(0, bytes, 0, bytes.length);
          dev.copyFromHostBuffer(host);
        }
        out[i] = dev;
      }
      return out;
    } catch (Throwable t) {
      closeAll(out);
      throw t;
    }
  }

  /** Best-effort close-all for an array of {@link DeviceMemoryBuffer}; ignores nulls. */
  public static void closeAll(DeviceMemoryBuffer[] buffers) {
    if (buffers == null) return;
    for (DeviceMemoryBuffer b : buffers) {
      if (b != null) b.close();
    }
  }
}
