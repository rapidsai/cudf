/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class represents data in some form on the GPU. Closing this object will effectively release
 * the memory held by the buffer.  Note that because of pooling in RMM or reference counting if a
 * buffer is sliced it may not actually result in the memory being released.
 */
public class DeviceMemoryBuffer extends BaseDeviceMemoryBuffer {
  private static final Logger log = LoggerFactory.getLogger(DeviceMemoryBuffer.class);

  private static final class DeviceBufferCleaner extends MemoryBufferCleaner {
    private long address;
    private long lengthInBytes;
    private Cuda.Stream stream;

    DeviceBufferCleaner(long address, long lengthInBytes, Cuda.Stream stream) {
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
          Rmm.free(address, lengthInBytes, s);
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
        log.error("A DEVICE BUFFER WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked device buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return address == 0;
    }
  }

  private static final class RmmDeviceBufferCleaner extends MemoryBufferCleaner {
    private long rmmBufferAddress;

    RmmDeviceBufferCleaner(long rmmBufferAddress) {
      this.rmmBufferAddress = rmmBufferAddress;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (rmmBufferAddress != 0) {
        Rmm.freeDeviceBuffer(rmmBufferAddress);
        rmmBufferAddress = 0;
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("WE LEAKED A DEVICE BUFFER!!!!");
        logRefCountDebug("Leaked device buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return rmmBufferAddress == 0;
    }
  }

  /**
   * Wrap an existing RMM allocation in a device memory buffer. The RMM allocation will be freed
   * when the resulting device memory buffer instance frees its memory resource (i.e.: when its
   * reference count goes to zero).
   * @param address device address of the RMM allocation
   * @param lengthInBytes length of the RMM allocation in bytes
   * @param rmmBufferAddress host address of the rmm::device_buffer that owns the device memory
   * @return new device memory buffer instance that wraps the existing RMM allocation
   */
  public static DeviceMemoryBuffer fromRmm(long address, long lengthInBytes, long rmmBufferAddress) {
    return new DeviceMemoryBuffer(address, lengthInBytes, rmmBufferAddress);
  }

  DeviceMemoryBuffer(long address, long lengthInBytes, MemoryBufferCleaner cleaner) {
    super(address, lengthInBytes, cleaner);
  }

  DeviceMemoryBuffer(long address, long lengthInBytes, long rmmBufferAddress) {
    super(address, lengthInBytes, new RmmDeviceBufferCleaner(rmmBufferAddress));
  }

  DeviceMemoryBuffer(long address, long lengthInBytes, Cuda.Stream stream) {
    super(address, lengthInBytes, new DeviceBufferCleaner(address, lengthInBytes, stream));
  }

  private DeviceMemoryBuffer(long address, long lengthInBytes, DeviceMemoryBuffer parent) {
    super(address, lengthInBytes, parent);
  }

  /**
   * Allocate memory for use on the GPU. You must close it when done.
   * @param bytes size in bytes to allocate
   * @return the buffer
   */
  public static DeviceMemoryBuffer allocate(long bytes) {
    return allocate(bytes, Cuda.DEFAULT_STREAM);
  }

  /**
   * Allocate memory for use on the GPU. You must close it when done.
   * @param bytes size in bytes to allocate
   * @param stream The stream in which to synchronize this command
   * @return the buffer
   */
  public static DeviceMemoryBuffer allocate(long bytes, Cuda.Stream stream) {
    return Rmm.alloc(bytes, stream);
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
  public synchronized final DeviceMemoryBuffer slice(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");
    incRefCount();
    return new DeviceMemoryBuffer(getAddress() + offset, len, this);
  }

  /**
   * Convert a view that is a subset of this Buffer by slicing this.
   * @param view the view to use as a reference.
   * @return the sliced buffer.
   */
  synchronized final BaseDeviceMemoryBuffer sliceFrom(DeviceMemoryBufferView view) {
    if (view == null) {
      return null;
    }
    addressOutOfBoundsCheck(view.address, view.length, "sliceFrom");
    incRefCount();
    return new DeviceMemoryBuffer(view.address, view.length, this);
  }
}
