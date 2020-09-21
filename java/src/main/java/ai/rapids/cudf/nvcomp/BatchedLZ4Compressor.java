/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.rapids.cudf.nvcomp;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;

public class BatchedLZ4Compressor {
  public static class BatchedCompressionResult {
    private final DeviceMemoryBuffer[] compressedBuffers;
    private final long[] compressedSizes;

    BatchedCompressionResult(DeviceMemoryBuffer[] buffers, long[] sizes) {
      this.compressedBuffers = buffers;
      this.compressedSizes = sizes;
    }

    public DeviceMemoryBuffer[] getCompressedBuffers() {
      return compressedBuffers;
    }

    public long[] getCompressedSizes() {
      return compressedSizes;
    }
  }

  public static long getTempSize(BaseDeviceMemoryBuffer[] inputs, long chunkSize) {
    int numBuffers = inputs.length;
    long[] inputAddrs = new long[numBuffers];
    long[] inputSizes = new long[numBuffers];
    for (int i = 0; i < numBuffers; ++i) {
      BaseDeviceMemoryBuffer buffer = inputs[i];
      inputAddrs[i] = buffer.getAddress();
      inputSizes[i] = buffer.getLength();
    }
    return NvcompJni.batchedLZ4CompressGetTempSize(inputAddrs, inputSizes, chunkSize);
  }

  public static long[] getOutputSizes(BaseDeviceMemoryBuffer[] inputs, long chunkSize,
                                      BaseDeviceMemoryBuffer tempBuffer) {
    int numBuffers = inputs.length;
    long[] inputAddrs = new long[numBuffers];
    long[] inputSizes = new long[numBuffers];
    for (int i = 0; i < numBuffers; ++i) {
      BaseDeviceMemoryBuffer buffer = inputs[i];
      inputAddrs[i] = buffer.getAddress();
      inputSizes[i] = buffer.getLength();
    }
    return NvcompJni.batchedLZ4CompressGetOutputSize(inputAddrs, inputSizes, chunkSize,
        tempBuffer.getAddress(), tempBuffer.getLength());
  }

  public static void compressAsync(HostMemoryBuffer compressedSizesOutputBuffer,
                                   BaseDeviceMemoryBuffer[] inputs, long chunkSize,
                                   BaseDeviceMemoryBuffer tempBuffer,
                                   BaseDeviceMemoryBuffer[] outputs, Cuda.Stream stream) {
    int numBuffers = inputs.length;
    if (outputs.length != numBuffers) {
      throw new IllegalArgumentException("buffer count mismatch, " + numBuffers + " inputs and " +
          outputs.length + " outputs");
    }
    if (compressedSizesOutputBuffer.getLength() < numBuffers * 8) {
      throw new IllegalArgumentException("compressed output size buffer must be able to hold " +
          "at least 8 bytes per buffer, size is only " + compressedSizesOutputBuffer.getLength());
    }

    long[] inputAddrs = new long[numBuffers];
    long[] inputSizes = new long[numBuffers];
    for (int i = 0; i < numBuffers; ++i) {
      BaseDeviceMemoryBuffer buffer = inputs[i];
      inputAddrs[i] = buffer.getAddress();
      inputSizes[i] = buffer.getLength();
    }

    long[] outputAddrs = new long[numBuffers];
    long[] outputSizes = new long[numBuffers];
    for (int i = 0; i < numBuffers; ++i) {
      BaseDeviceMemoryBuffer buffer = outputs[i];
      outputAddrs[i] = buffer.getAddress();
      outputSizes[i] = buffer.getLength();
    }

    NvcompJni.batchedLZ4CompressAsync(compressedSizesOutputBuffer.getAddress(),
        inputAddrs, inputSizes, chunkSize, tempBuffer.getAddress(), tempBuffer.getLength(),
        outputAddrs, outputSizes, stream.getStream());
  }

  public static BatchedCompressionResult compress(BaseDeviceMemoryBuffer[] inputs, long chunkSize,
                                                  Cuda.Stream stream) {
    int numBuffers = inputs.length;
    long[] inputAddrs = new long[numBuffers];
    long[] inputSizes = new long[numBuffers];
    for (int i = 0; i < numBuffers; ++i) {
      BaseDeviceMemoryBuffer buffer = inputs[i];
      inputAddrs[i] = buffer.getAddress();
      inputSizes[i] = buffer.getLength();
    }

    DeviceMemoryBuffer[] outputBuffers = new DeviceMemoryBuffer[numBuffers];
    try {
      long tempSize = NvcompJni.batchedLZ4CompressGetTempSize(inputAddrs, inputSizes, chunkSize);
      try (DeviceMemoryBuffer tempBuffer = DeviceMemoryBuffer.allocate(tempSize)) {
        long[] outputSizes = NvcompJni.batchedLZ4CompressGetOutputSize(inputAddrs, inputSizes,
                chunkSize, tempBuffer.getAddress(), tempBuffer.getLength());
        long[] outputAddrs = new long[numBuffers];
        for (int i = 0; i < numBuffers; ++i) {
          DeviceMemoryBuffer buffer = DeviceMemoryBuffer.allocate(outputSizes[i]);
          outputBuffers[i] = buffer;
          outputAddrs[i] = buffer.getAddress();
        }

        try (HostMemoryBuffer compressedSizesBuffer = HostMemoryBuffer.allocate(numBuffers * 8)) {
          NvcompJni.batchedLZ4CompressAsync(compressedSizesBuffer.getAddress(),
                  inputAddrs, inputSizes, chunkSize,
                  tempBuffer.getAddress(), tempBuffer.getLength(),
                  outputAddrs, outputSizes, stream.getStream());
          stream.sync();
          long[] compressedSizes = new long[numBuffers];
          compressedSizesBuffer.getLongs(compressedSizes, 0, 0, numBuffers);
          return new BatchedCompressionResult(outputBuffers, compressedSizes);
        }
      }
    } catch (Throwable t) {
      for (DeviceMemoryBuffer buffer : outputBuffers) {
        if (buffer != null) {
          buffer.close();
        }
      }
      throw t;
    }
  }
}
