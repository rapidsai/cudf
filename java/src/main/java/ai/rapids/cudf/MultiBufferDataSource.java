/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * This is a DataSource that can take multiple HostMemoryBuffers. They
 * are treated as if they are all part of a single file connected end to end.
 */
public class MultiBufferDataSource extends DataSource {
  private final long sizeInBytes;
  private final HostMemoryBuffer[] hostBuffers;
  private final long[] startOffsets;
  private final HostMemoryAllocator allocator;

  // Metrics
  private long hostReads = 0;
  private long hostReadBytes = 0;
  private long devReads = 0;
  private long devReadBytes = 0;

  /**
   * Create a new data source backed by multiple buffers.
   * @param buffers the buffers that will back the data source.
   */
  public MultiBufferDataSource(HostMemoryBuffer ... buffers) {
    this(DefaultHostMemoryAllocator.get(), buffers);
  }

  /**
   * Create a new data source backed by multiple buffers.
   * @param allocator the allocator to use for host buffers, if needed.
   * @param buffers the buffers that will back the data source.
   */
  public MultiBufferDataSource(HostMemoryAllocator allocator, HostMemoryBuffer ... buffers) {
    int numBuffers = buffers.length;
    hostBuffers = new HostMemoryBuffer[numBuffers];
    startOffsets = new long[numBuffers];

    long currentOffset = 0;
    for (int i = 0; i < numBuffers; i++) {
      HostMemoryBuffer hmb = buffers[i];
      hmb.incRefCount();
      hostBuffers[i] = hmb;
      startOffsets[i] = currentOffset;
      currentOffset += hmb.getLength();
    }
    sizeInBytes = currentOffset;
    this.allocator = allocator;
  }

  @Override
  public long size() {
    return sizeInBytes;
  }

  private int getStartBufferIndexForOffset(long offset) {
    assert (offset >= 0);

    // It is super common to read from the start or end of a file (the header or footer)
    // so special case them
    if (offset == 0) {
      return 0;
    }
    int startIndex = 0;
    int endIndex = startOffsets.length - 1;
    if (offset >= startOffsets[endIndex]) {
      return endIndex;
    }
    while (startIndex != endIndex) {
      int midIndex = (int)(((long)startIndex + endIndex) / 2);
      long midStartOffset = startOffsets[midIndex];
      if (offset >= midStartOffset) {
        // It is either in mid or after mid.
        if (midIndex == endIndex || offset <= startOffsets[midIndex + 1]) {
          // We found it in mid
          return midIndex;
        } else {
          // It is after mid
          startIndex = midIndex + 1;
        }
      } else {
        // It is before mid
        endIndex = midIndex - 1;
      }
    }
    return startIndex;
  }


  interface DoCopy<T extends MemoryBuffer> {
    void copyFromHostBuffer(T dest, long destOffset, HostMemoryBuffer src,
                            long srcOffset, long srcAmount);
  }

  private <T extends MemoryBuffer> long read(long offset, T dest, DoCopy<T> doCopy) {
    assert (offset >= 0);
    long realOffset = Math.min(offset, sizeInBytes);
    long realAmount = Math.min(sizeInBytes - realOffset, dest.getLength());

    int index = getStartBufferIndexForOffset(realOffset);

    HostMemoryBuffer buffer = hostBuffers[index];
    long bufferOffset = realOffset - startOffsets[index];
    long bufferAmount = Math.min(buffer.length - bufferOffset, realAmount);
    long remainingAmount = realAmount;
    long currentOffset = realOffset;
    long outputOffset = 0;

    while (remainingAmount > 0) {
      doCopy.copyFromHostBuffer(dest, outputOffset, buffer,
          bufferOffset, bufferAmount);
      remainingAmount -= bufferAmount;
      outputOffset += bufferAmount;
      currentOffset += bufferAmount;
      index++;
      if (index < hostBuffers.length) {
        buffer = hostBuffers[index];
        bufferOffset = currentOffset - startOffsets[index];
        bufferAmount = Math.min(buffer.length - bufferOffset, remainingAmount);
      }
    }

    return realAmount;
  }

  @Override
  public HostMemoryBuffer hostRead(long offset, long amount) {
    assert (offset >= 0);
    assert (amount >= 0);
    long realOffset = Math.min(offset, sizeInBytes);
    long realAmount = Math.min(sizeInBytes - realOffset, amount);

    int index = getStartBufferIndexForOffset(realOffset);

    HostMemoryBuffer buffer = hostBuffers[index];
    long bufferOffset = realOffset - startOffsets[index];
    long bufferAmount = Math.min(buffer.length - bufferOffset, realAmount);
    if (bufferAmount == realAmount) {
      hostReads += 1;
      hostReadBytes += realAmount;
      // It all fits in a single buffer, so do a zero copy operation
      return buffer.slice(bufferOffset, bufferAmount);
    } else {
      // We will have to allocate a new buffer and copy data into it.
      boolean success = false;
      HostMemoryBuffer ret = allocator.allocate(realAmount, true);
      try {
        long amountRead = read(offset, ret, HostMemoryBuffer::copyFromHostBuffer);
        assert(amountRead == realAmount);
        hostReads += 1;
        hostReadBytes += amountRead;
        success = true;
        return ret;
      } finally {
        if (!success) {
          ret.close();
        }
      }
    }
  }

  @Override
  public long hostRead(long offset, HostMemoryBuffer dest) {
    long ret = read(offset, dest, HostMemoryBuffer::copyFromHostBuffer);
    hostReads += 1;
    hostReadBytes += ret;
    return ret;
  }

  @Override
  public boolean supportsDeviceRead() {
    return true;
  }

  @Override
  public long deviceRead(long offset, DeviceMemoryBuffer dest,
                         Cuda.Stream stream) {
    long ret = read(offset, dest, (destParam, destOffset, src, srcOffset, srcAmount) ->
        destParam.copyFromHostBufferAsync(destOffset, src, srcOffset, srcAmount, stream));
    devReads += 1;
    devReadBytes += ret;
    return ret;
  }


  @Override
  public void close() {
    try {
      super.close();
    } finally {
      for (HostMemoryBuffer hmb: hostBuffers) {
        if (hmb != null) {
          hmb.close();
        }
      }
    }
  }

  public long getHostReads() {
    return hostReads;
  }

  public long getHostReadBytes() {
    return hostReadBytes;
  }

  public long getDevReads() {
    return devReads;
  }

  public long getDevReadBytes() {
    return devReadBytes;
  }
}
