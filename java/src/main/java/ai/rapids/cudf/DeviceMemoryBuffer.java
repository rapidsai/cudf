/*
 *
 *  Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
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
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
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
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
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

  // Static factory method to make this a little simpler from JNI
  static DeviceMemoryBuffer fromRmm(long address, long lengthInBytes, long rmmBufferAddress) {
    return new DeviceMemoryBuffer(address, lengthInBytes, rmmBufferAddress);
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
    return Rmm.alloc(bytes);
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
