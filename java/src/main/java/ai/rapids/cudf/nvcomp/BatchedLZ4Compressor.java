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

/** Multi-buffer LZ4 compressor */
public class BatchedLZ4Compressor {
  /** Describes a batched compression result */
  public static class BatchedCompressionResult {
    private final DeviceMemoryBuffer[] compressedBuffers;
    private final long[] compressedSizes;

    BatchedCompressionResult(DeviceMemoryBuffer[] buffers, long[] sizes) {
      this.compressedBuffers = buffers;
      this.compressedSizes = sizes;
    }

    /**
     * Get the output compressed buffers corresponding to the input buffers.
     * Note that the buffers are likely larger than required to store the compressed data.
     */
    public DeviceMemoryBuffer[] getCompressedBuffers() {
      return compressedBuffers;
    }

    /** Get the corresponding amount of compressed data in each output buffer. */
    public long[] getCompressedSizes() {
      return compressedSizes;
    }
  }

  /**
   * Get the amount of temporary storage space required to compress a batch of buffers.
   * @param inputs    batch of data buffers to be individually compressed
   * @param chunkSize compression chunk size to use
   * @return amount in bytes of temporary storage space required to compress the batch
   */
  public static long getTempSize(BaseDeviceMemoryBuffer[] inputs, long chunkSize) {
    if (chunkSize <= 0) {
      throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
    }
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

  /**
   * Get the amount of output storage space required to compress a batch of buffers.
   * @param inputs     batch of data buffers to be individually compressed
   * @param chunkSize  compression chunk size to use
   * @param tempBuffer temporary storage space
   * @return amount in bytes of output storage space corresponding to each input buffer in the batch
   */
  public static long[] getOutputSizes(BaseDeviceMemoryBuffer[] inputs, long chunkSize,
                                      BaseDeviceMemoryBuffer tempBuffer) {
    if (chunkSize <= 0) {
      throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
    }
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

  /**
   * Calculates the minimum size in bytes necessary to store the compressed output sizes
   * when performing an asynchronous batch compression.
   * @param numBuffers number of buffers in the batch
   * @return minimum size of the compressed output sizes buffer needed
   */
  public static long getCompressedSizesBufferSize(int numBuffers) {
    // Each compressed size value is a 64-bit long
    return numBuffers * 8;
  }

  /**
   * Asynchronously compress a batch of input buffers. The compressed size output buffer must be
   * pinned memory for this operation to be truly asynchronous. Note that the caller must
   * synchronize on the specified CUDA stream in order to safely examine the compressed output
   * sizes!
   * @param compressedSizesOutputBuffer host memory where the compressed output size will be stored
   * @param inputs     buffers to compress
   * @param chunkSize  type of data within each buffer
   * @param tempBuffer compression chunk size to use
   * @param outputs    output buffers that will contain the compressed results
   * @param stream     CUDA stream to use
   */
  public static void compressAsync(HostMemoryBuffer compressedSizesOutputBuffer,
                                   BaseDeviceMemoryBuffer[] inputs, long chunkSize,
                                   BaseDeviceMemoryBuffer tempBuffer,
                                   BaseDeviceMemoryBuffer[] outputs, Cuda.Stream stream) {
    if (chunkSize <= 0) {
      throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
    }
    int numBuffers = inputs.length;
    if (outputs.length != numBuffers) {
      throw new IllegalArgumentException("buffer count mismatch, " + numBuffers + " inputs and " +
          outputs.length + " outputs");
    }
    if (compressedSizesOutputBuffer.getLength() < getCompressedSizesBufferSize(numBuffers)) {
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

  /**
   * Compress a batch of buffers with LZ4
   * @param inputs    buffers to compress
   * @param chunkSize compression chunk size to use
   * @param stream    CUDA stream to use
   * @return compression results containing the corresponding output buffer and compressed data size
   *         for each input buffer
   */
  public static BatchedCompressionResult compress(BaseDeviceMemoryBuffer[] inputs, long chunkSize,
                                                  Cuda.Stream stream) {
    if (chunkSize <= 0) {
      throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
    }
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

        long compressedSizesBufferSize = getCompressedSizesBufferSize(numBuffers);
        try (HostMemoryBuffer compressedSizesBuffer =
                 HostMemoryBuffer.allocate(compressedSizesBufferSize)) {
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
