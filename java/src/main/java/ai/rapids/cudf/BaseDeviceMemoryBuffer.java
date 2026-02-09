/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Base class for all MemoryBuffers that are in device memory.
 */
public abstract class BaseDeviceMemoryBuffer extends MemoryBuffer {
  protected BaseDeviceMemoryBuffer(long address, long length, MemoryBuffer parent) {
    super(address, length, parent);
  }

  protected BaseDeviceMemoryBuffer(long address, long length, MemoryBufferCleaner cleaner) {
    super(address, length, cleaner);
  }

  /**
   * Copy a subset of src to this buffer starting at destOffset.
   * @param destOffset the offset in this to start copying from.
   * @param src what to copy from
   * @param srcOffset offset into src to start out
   * @param length how many bytes to copy
   */
  public final void copyFromHostBuffer(long destOffset, HostMemoryBuffer src, long srcOffset, long length) {
    addressOutOfBoundsCheck(address + destOffset, length, "copy range dest");
    src.addressOutOfBoundsCheck(src.address + srcOffset, length, "copy range src");
    Cuda.memcpy(address + destOffset, src.address + srcOffset, length, CudaMemcpyKind.HOST_TO_DEVICE);
  }

  /**
   * Copy a subset of src to this buffer starting at destOffset using the specified CUDA stream.
   * The copy has completed when this returns, but the memory copy could overlap with
   * operations occurring on other streams.
   * @param destOffset the offset in this to start copying from.
   * @param src what to copy from
   * @param srcOffset offset into src to start out
   * @param length how many bytes to copy
   * @param stream CUDA stream to use
   */
  public final void copyFromHostBuffer(long destOffset, HostMemoryBuffer src,
      long srcOffset, long length, Cuda.Stream stream) {
    addressOutOfBoundsCheck(address + destOffset, length, "copy range dest");
    src.addressOutOfBoundsCheck(src.address + srcOffset, length, "copy range src");
    Cuda.memcpy(address + destOffset, src.address + srcOffset, length,
        CudaMemcpyKind.HOST_TO_DEVICE, stream);
  }

  /**
   * Copy a subset of src to this buffer starting at destOffset using the specified CUDA stream.
   * The copy is async and may not have completed when this returns.
   * @param destOffset the offset in this to start copying from.
   * @param src what to copy from
   * @param srcOffset offset into src to start out
   * @param length how many bytes to copy
   * @param stream CUDA stream to use
   */
  public final void copyFromHostBufferAsync(long destOffset, HostMemoryBuffer src,
                                            long srcOffset, long length, Cuda.Stream stream) {
    addressOutOfBoundsCheck(address + destOffset, length, "copy range dest");
    src.addressOutOfBoundsCheck(src.address + srcOffset, length, "copy range src");
    Cuda.asyncMemcpy(address + destOffset, src.address + srcOffset, length,
        CudaMemcpyKind.HOST_TO_DEVICE, stream);
  }

  /**
   * Copy a subset of src to this buffer starting at destOffset using the specified CUDA stream.
   * The copy is async and may not have completed when this returns.
   * @param destOffset the offset in this to start copying from.
   * @param src what to copy from
   * @param srcOffset offset into src to start out
   * @param length how many bytes to copy
   * @param stream CUDA stream to use
   */
  public final void copyFromDeviceBufferAsync(long destOffset, BaseDeviceMemoryBuffer src,
                                              long srcOffset, long length, Cuda.Stream stream) {
    addressOutOfBoundsCheck(address + destOffset, length, "copy range dest");
    src.addressOutOfBoundsCheck(src.address + srcOffset, length, "copy range src");
    Cuda.asyncMemcpy(address + destOffset, src.address + srcOffset, length,
        CudaMemcpyKind.DEVICE_TO_DEVICE, stream);
  }

  /**
   * Copy a subset of src to this buffer starting at the beginning of this.
   * @param src what to copy from
   * @param srcOffset offset into src to start out
   * @param length how many bytes to copy
   */
  public final void copyFromHostBuffer(HostMemoryBuffer src, long srcOffset, long length) {
    copyFromHostBuffer(0, src, srcOffset, length);
  }

  /**
   * Copy everything from src to this buffer starting at the beginning of this buffer.
   * @param src - Buffer to copy data from
   */
  public final void copyFromHostBuffer(HostMemoryBuffer src) {
    copyFromHostBuffer(0, src, 0, src.length);
  }

  /**
   * Copy entire host buffer starting at the beginning of this buffer using a CUDA stream.
   * The copy has completed when this returns, but the memory copy could overlap with
   * operations occurring on other streams.
   * @param src host buffer to copy from
   * @param stream CUDA stream to use
   */
  public final void copyFromHostBuffer(HostMemoryBuffer src, Cuda.Stream stream) {
    copyFromHostBuffer(0, src, 0, src.length, stream);
  }

  /**
   * Copy entire host buffer starting at the beginning of this buffer using a CUDA stream.
   * The copy is async and may not have completed when this returns.
   * @param src host buffer to copy from
   * @param stream CUDA stream to use
   */
  public final void copyFromHostBufferAsync(HostMemoryBuffer src, Cuda.Stream stream) {
    copyFromHostBufferAsync(0, src, 0, src.length, stream);
  }

  /**
   * Slice off a part of the device buffer, copying it instead of reference counting it.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a device buffer that will need to be closed independently from this buffer.
   */
  public final DeviceMemoryBuffer sliceWithCopy(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");
    DeviceMemoryBuffer ret = null;
    boolean success = false;
    try {
      ret = DeviceMemoryBuffer.allocate(len);
      Cuda.memcpy(ret.getAddress(), getAddress() + offset, len, CudaMemcpyKind.DEVICE_TO_DEVICE);
      success = true;
      return ret;
    } finally {
      if (!success && ret != null) {
        ret.close();
      }
    }
  }
}
