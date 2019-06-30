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
 * This class represents a Address held in the GPU
 */
class DeviceMemoryBuffer extends MemoryBuffer {

  DeviceMemoryBuffer(long address, long lengthInBytes) {
    super(address, lengthInBytes);
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
  public void copyFromHostBuffer(HostMemoryBuffer hostBuffer) {
    addressOutOfBoundsCheck(address, hostBuffer.length, "copy range dest");
    assert !hostBuffer.closed;
    Cuda.memcpy(address, hostBuffer.address, hostBuffer.length, CudaMemcpyKind.HOST_TO_DEVICE);
  }

  /**
   * Close this Buffer and free memory allocated
   */
  @Override
  protected void doClose() {
    Rmm.free(address, 0);
  }

}
