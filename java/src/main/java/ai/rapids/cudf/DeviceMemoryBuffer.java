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
 * This class represents data in some form on the GPU. Closing this object will effectively release
 * the memory held by the buffer.  Note that because of pooling in RMM or reference counting if a
 * buffer is sliced it may not actually result in the memory being released.
 */
public class DeviceMemoryBuffer extends MemoryBuffer {
  private static final Logger log = LoggerFactory.getLogger(DeviceMemoryBuffer.class);

  private static final class DeviceBufferCleaner extends MemoryBufferCleaner {
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
   * Allocate memory for use on the GPU. You must close it when done.
   * @param bytes size in bytes to allocate
   * @return the buffer
   */
  public static DeviceMemoryBuffer allocate(long bytes) {
    return new DeviceMemoryBuffer(Rmm.alloc(bytes, 0), bytes);
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
   * Slice off a part of the device buffer. Note that this is a zero copy operation and all
   * slices must be closed along with the original buffer before the memory is released to RMM.
   * So use this with some caution.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a device buffer that will need to be closed independently from this buffer.
   */
  public final DeviceMemoryBuffer slice(long offset, long len) {
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
  public final DeviceMemoryBuffer sliceWithCopy(long offset, long len) {
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