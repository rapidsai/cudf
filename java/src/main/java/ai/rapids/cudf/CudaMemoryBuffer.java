/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class represents data allocated using `cudaMalloc` directly instead of the default RMM
 * memory resource. Closing this object will effectively release the memory held by the buffer.
 * Note that because of reference counting if a buffer is sliced it may not actually result in the
 * memory being released.
 */
public class CudaMemoryBuffer extends BaseDeviceMemoryBuffer {
  private static final Logger log = LoggerFactory.getLogger(CudaMemoryBuffer.class);

  private static final class CudaBufferCleaner extends MemoryBufferCleaner {
    private long address;
    private long lengthInBytes;
    private Cuda.Stream stream;

    CudaBufferCleaner(long address, long lengthInBytes, Cuda.Stream stream) {
      this.address = address;
      this.lengthInBytes = lengthInBytes;
      this.stream = stream;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = address;
      if (address != 0) {
        long s = stream == null ? 0 : stream.getStream();
        try {
          Rmm.freeCuda(address, lengthInBytes, s);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          address = 0;
          lengthInBytes = 0;
          stream = null;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A CUDA BUFFER WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked device buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return address == 0;
    }
  }

  /**
   * Wrap an existing CUDA allocation in a device memory buffer. The CUDA allocation will be freed
   * when the resulting device memory buffer instance frees its memory resource (i.e.: when its
   * reference count goes to zero).
   * @param address device address of the CUDA memory allocation
   * @param lengthInBytes length of the CUDA allocation in bytes
   * @param stream CUDA stream to use for synchronization when freeing the allocation
   */
  public CudaMemoryBuffer(long address, long lengthInBytes, Cuda.Stream stream) {
    super(address, lengthInBytes, new CudaBufferCleaner(address, lengthInBytes, stream));
  }

  private CudaMemoryBuffer(long address, long lengthInBytes, CudaMemoryBuffer parent) {
    super(address, lengthInBytes, parent);
  }

  /**
   * Allocate memory for use on the GPU. You must close it when done.
   * @param bytes size in bytes to allocate
   * @return the buffer
   */
  public static CudaMemoryBuffer allocate(long bytes) {
    return allocate(bytes, Cuda.DEFAULT_STREAM);
  }

  /**
   * Allocate memory for use on the GPU. You must close it when done.
   * @param bytes size in bytes to allocate
   * @param stream The stream in which to synchronize this command
   * @return the buffer
   */
  public static CudaMemoryBuffer allocate(long bytes, Cuda.Stream stream) {
    return Rmm.allocCuda(bytes, stream);
  }

  /**
   * Slice off a part of the device buffer. Note that this is a zero copy operation and all
   * slices must be closed along with the original buffer before the memory is released to RMM.
   * So use this with some caution.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a device buffer that will need to be closed independently from this buffer.
   */
  @Override
  public synchronized final CudaMemoryBuffer slice(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");
    incRefCount();
    return new CudaMemoryBuffer(getAddress() + offset, len, this);
  }
}
