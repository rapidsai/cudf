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

/**
 * Base class for all MemoryBuffers that are in device memory.
 */
public class BaseDeviceMemoryBuffer extends MemoryBuffer {
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
