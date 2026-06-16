/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Represents a cuFile buffer.
 */
public final class CuFileBuffer extends BaseDeviceMemoryBuffer {
  private static final int ALIGNMENT = 4096;
  private final DeviceMemoryBuffer deviceMemoryBuffer;
  private final CuFileResourceCleaner cleaner;

  static {
    CuFile.initialize();
  }

  /**
   * Construct a new cuFile buffer.
   *
   * @param buffer         The device memory buffer used for the cuFile buffer. This buffer is owned
   *                       by the cuFile buffer, and will be closed when the cuFile buffer is closed.
   * @param registerBuffer If true, register the cuFile buffer.
   */
  private CuFileBuffer(DeviceMemoryBuffer buffer, boolean registerBuffer) {
    super(buffer.address, buffer.length, (MemoryBufferCleaner) null);
    if (registerBuffer && !isAligned(buffer)) {
      buffer.close();
      throw new IllegalArgumentException(
          "To register a cuFile buffer, its length must be a multiple of " + ALIGNMENT);
    }
    deviceMemoryBuffer = buffer;
    cleaner = new CuFileResourceCleaner(create(buffer.address, buffer.length, registerBuffer), CuFileBuffer::destroy);
    MemoryCleaner.register(this, cleaner);
  }

  /**
   * Allocate memory for use with cuFile on the GPU. You must close it when done.
   *
   * @param bytes          size in bytes to allocate
   * @param registerBuffer If true, register the cuFile buffer.
   * @return the buffer
   */
  public static CuFileBuffer allocate(long bytes, boolean registerBuffer) {
    DeviceMemoryBuffer buffer = DeviceMemoryBuffer.allocate(bytes);
    return new CuFileBuffer(buffer, registerBuffer);
  }

  @Override
  public MemoryBuffer slice(long offset, long len) {
    throw new UnsupportedOperationException("Slice on cuFile buffer is not supported");
  }

  @Override
  public void close() {
    cleaner.close(this);
    deviceMemoryBuffer.close();
  }

  long getPointer() {
    return cleaner.getPointer();
  }

  private boolean isAligned(BaseDeviceMemoryBuffer buffer) {
    return buffer.length % ALIGNMENT == 0;
  }

  private static native long create(long address, long length, boolean registerBuffer);

  private static native void destroy(long pointer);
}
