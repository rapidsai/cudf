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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Base class for all MemoryBuffers that are in device memory.
 */
public abstract class BaseDeviceMemoryBuffer extends MemoryBuffer {

  private static final Logger log = LoggerFactory.getLogger(BaseDeviceMemoryBuffer.class);

  private static final boolean BOOKKEEP_MEMORY = Boolean.getBoolean("ai.rapids.memory.bookkeep");
  private static final ConcurrentHashMap<Long, LongAdder> deviceMemPerThread = new ConcurrentHashMap<>();
  private static final ConcurrentHashMap<Long, Long> addr2threadId = new ConcurrentHashMap<>();

  private void bookkeepDeviceMemory(long threadId, long amount) {
    LongAdder adder = deviceMemPerThread.computeIfAbsent(threadId, key -> new LongAdder());
    adder.add(amount);
  }

  // Called from the application to get a summary of the device memory bookkeeping
  public static String getDeviceMemoryBookkeepSummary() {
    StringBuilder sb = new StringBuilder();
    sb.append("<<Device Memory Bookkeeping>>\n");
    for (Map.Entry<Long, LongAdder> entry : deviceMemPerThread.entrySet()) {
      long threadId = entry.getKey();
      LongAdder adder = entry.getValue();
      sb.append("Thread with ID ").append(threadId).append(" is accountable for ")
          .append(adder.sum()).append(" bytes\n");
    }
    return sb.toString();
  }

  public static class MemoryBookkeeper implements EventHandler {
    private long ptr;
    private long amount;

    public MemoryBookkeeper(long ptr, long amount) {
      this.ptr = ptr;
      this.amount = amount;
    }

    @Override
    public void onClosed(int refCount) {
      if (refCount == 0) {
        if (BOOKKEEP_MEMORY) {
          Long threadId = addr2threadId.get(ptr);
          if (threadId != null) {
            LongAdder adder = deviceMemPerThread.get(threadId);
            if (adder != null) {
              adder.add(-amount);
            } else {
              log.warn("Could not find adder for thread {} from address {} bytes: {}",
                  threadId, ptr, amount);
            }
            addr2threadId.remove(ptr);
          } else {
            log.warn("Could not find thread id for address {} bytes: {}", ptr, amount);
          }
        }
      }
    }
  }

  protected BaseDeviceMemoryBuffer(long address, long length, MemoryBuffer parent) {
    super(address, length, parent);
  }

  protected BaseDeviceMemoryBuffer(long address, long length, MemoryBufferCleaner cleaner) {
    super(address, length, cleaner);

    if (BOOKKEEP_MEMORY && cleaner != null && !(cleaner instanceof SlicedBufferCleaner)) {
      long threadId = Thread.currentThread().getId();
      addr2threadId.put(address, threadId);
      bookkeepDeviceMemory(threadId, length);
      setEventHandler(new MemoryBookkeeper(address, length));
    }
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
