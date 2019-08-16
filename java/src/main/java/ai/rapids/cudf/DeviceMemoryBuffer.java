/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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
 * This class represents a Address held in the GPU
 */
class DeviceMemoryBuffer extends MemoryBuffer {
  private static final Logger log = LoggerFactory.getLogger(DeviceMemoryBuffer.class);

  private static final class DeviceBufferCleaner extends MemoryCleaner.Cleaner {
    private long address;

    DeviceBufferCleaner(long address) {
      this.address = address;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (address != 0) {
        Rmm.free(address, 0);
        address = 0;
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("WE LEAKED A DEVICE BUFFER!!!!");
        logRefCountDebug("Leaked device buffer");
      }
      return neededCleanup;
    }
  }

  DeviceMemoryBuffer(long address, long lengthInBytes) {
    super(address, lengthInBytes, new DeviceBufferCleaner(address));
  }

  private DeviceMemoryBuffer(long address, long lengthInBytes, DeviceMemoryBuffer parent) {
    super(address, lengthInBytes, parent);
  }

  /**
   * Factory method to create this buffer
   * @param bytes - size in bytes to allocate
   * @return - return this newly created buffer
   */
  public static DeviceMemoryBuffer allocate(long bytes) {
    return new DeviceMemoryBuffer(Rmm.alloc(bytes, 0), bytes);
  }

  /**
   * Method to copy from a HostMemoryBuffer to a DeviceMemoryBuffer
   * @param hostBuffer - Buffer to copy data from
   */
  final void copyFromHostBuffer(HostMemoryBuffer hostBuffer) {
    addressOutOfBoundsCheck(address, hostBuffer.length, "copy range dest");
    assert !hostBuffer.closed;
    Cuda.memcpy(address, hostBuffer.address, hostBuffer.length, CudaMemcpyKind.HOST_TO_DEVICE);
  }

  /**
   * Slice off a part of the device buffer.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a device buffer that will need to be closed independently from this buffer.
   */
  final DeviceMemoryBuffer slice(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");
    refCount++;
    cleaner.addRef();
    return new DeviceMemoryBuffer(getAddress() + offset, len, this);
  }

  /**
   * Slice off a part of the device buffer, copying it instead of reference counting it.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a device buffer that will need to be closed independently from this buffer.
   */
  final DeviceMemoryBuffer sliceWithCopy(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");
    DeviceMemoryBuffer ret = null;
    boolean success = false;
    try {
      ret = allocate(len);
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
