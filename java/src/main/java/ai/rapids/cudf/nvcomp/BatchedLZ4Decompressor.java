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

import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.MemoryCleaner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** LZ4 decompressor that operates on multiple input buffers in a batch */
public class BatchedLZ4Decompressor {
  private static final Logger log = LoggerFactory.getLogger(Decompressor.class);

  /**
   * Get the metadata associated with a batch of compressed buffers
   * @param inputs compressed buffers that will be decompressed
   * @param stream CUDA stream to use
   * @return opaque metadata object
   */
  public static BatchedMetadata getMetadata(BaseDeviceMemoryBuffer[] inputs, Cuda.Stream stream) {
    long[] inputAddrs = new long[inputs.length];
    long[] inputSizes = new long[inputs.length];
    for (int i = 0; i < inputs.length; ++i) {
      BaseDeviceMemoryBuffer buffer = inputs[i];
      inputAddrs[i] = buffer.getAddress();
      inputSizes[i] = buffer.getLength();
    }
    return new BatchedMetadata(NvcompJni.batchedLZ4DecompressGetMetadata(
        inputAddrs, inputSizes, stream.getStream()));
  }

  /**
   * Get the amount of temporary storage required to decompress a batch of buffers
   * @param metadata metadata retrieved from the compressed buffers
   * @return amount in bytes of temporary storage space required to decompress the buffer batch
   */
  public static long getTempSize(BatchedMetadata metadata) {
    return NvcompJni.batchedLZ4DecompressGetTempSize(metadata.getMetadata());
  }

  /**
   * Get the amount of ouptut storage required to decopmress a batch of buffers
   * @param metadata   metadata retrieved from the compressed buffers
   * @param numOutputs number of buffers in the batch
   * @return amount in bytes of temporary storage space required to decompress the buffer batch
   */
  public static long[] getOutputSizes(BatchedMetadata metadata, int numOutputs) {
    return NvcompJni.batchedLZ4DecompressGetOutputSize(metadata.getMetadata(), numOutputs);
  }

  /**
   * Asynchronously decompress a batch of buffers
   * @param inputs     buffers to decompress
   * @param tempBuffer temporary buffer
   * @param metadata   metadata retrieved from the compressed buffers
   * @param outputs    output buffers that will contain the compressed results
   * @param stream     CUDA stream to use
   */
  public static void decompressAsync(BaseDeviceMemoryBuffer[] inputs,
                                     BaseDeviceMemoryBuffer tempBuffer, BatchedMetadata metadata,
                                     BaseDeviceMemoryBuffer[] outputs, Cuda.Stream stream) {
    int numBuffers = inputs.length;
    if (outputs.length != numBuffers) {
      throw new IllegalArgumentException("buffer count mismatch, " + numBuffers + " inputs and " +
          outputs.length + " outputs");
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

    NvcompJni.batchedLZ4DecompressAsync(inputAddrs, inputSizes,
        tempBuffer.getAddress(), tempBuffer.getLength(), metadata.getMetadata(),
        outputAddrs, outputSizes, stream.getStream());
  }

  /**
   * Asynchronously decompress a batch of buffers
   * @param inputs buffers to decompress
   * @param stream CUDA stream to use
   * @return output buffers containing compressed data corresponding to the input buffers
   */
  public static DeviceMemoryBuffer[] decompressAsync(BaseDeviceMemoryBuffer[] inputs,
                                                     Cuda.Stream stream) {
    int numBuffers = inputs.length;
    long[] inputAddrs = new long[numBuffers];
    long[] inputSizes = new long[numBuffers];
    for (int i = 0; i < numBuffers; ++i) {
      BaseDeviceMemoryBuffer buffer = inputs[i];
      inputAddrs[i] = buffer.getAddress();
      inputSizes[i] = buffer.getLength();
    }

    long metadata = NvcompJni.batchedLZ4DecompressGetMetadata(inputAddrs, inputSizes,
            stream.getStream());
    try {
      long[] outputSizes = NvcompJni.batchedLZ4DecompressGetOutputSize(metadata, numBuffers);
      long[] outputAddrs = new long[numBuffers];
      DeviceMemoryBuffer[] outputs = new DeviceMemoryBuffer[numBuffers];
      try {
        for (int i = 0; i < numBuffers; ++i) {
          DeviceMemoryBuffer buffer = DeviceMemoryBuffer.allocate(outputSizes[i]);
          outputs[i] = buffer;
          outputAddrs[i] = buffer.getAddress();
        }

        long tempSize = NvcompJni.batchedLZ4DecompressGetTempSize(metadata);
        try (DeviceMemoryBuffer tempBuffer = DeviceMemoryBuffer.allocate(tempSize)) {
          NvcompJni.batchedLZ4DecompressAsync(inputAddrs, inputSizes,
                  tempBuffer.getAddress(), tempBuffer.getLength(), metadata,
                  outputAddrs, outputSizes, stream.getStream());
        }
      } catch (Throwable t) {
        for (DeviceMemoryBuffer buffer : outputs) {
          if (buffer != null) {
            buffer.close();
          }
        }
        throw t;
      }

      return outputs;
    } finally {
      NvcompJni.batchedLZ4DecompressDestroyMetadata(metadata);
    }
  }

  /** Opaque metadata object for batched LZ4 decompression */
  public static class BatchedMetadata implements AutoCloseable {
    private final BatchedMetadataCleaner cleaner;
    private final long id;
    private boolean closed = false;

    BatchedMetadata(long metadata) {
      this.cleaner = new BatchedMetadataCleaner(metadata);
      this.id = cleaner.id;
      MemoryCleaner.register(this, cleaner);
      cleaner.addRef();
    }

    long getMetadata() {
      return cleaner.metadata;
    }

    public boolean isLZ4Metadata() {
      return NvcompJni.isLZ4Metadata(getMetadata());
    }

    @Override
    public void close() {
      if (!closed) {
        cleaner.delRef();
        cleaner.clean(false);
        closed = true;
      } else {
        cleaner.logRefCountDebug("double free " + this);
        throw new IllegalStateException("Close called too many times " + this);
      }
    }

    @Override
    public String toString() {
      return "LZ4 BATCHED METADATA (ID: " + id + " " +
          Long.toHexString(cleaner.metadata) + ")";
    }

    private static class BatchedMetadataCleaner extends MemoryCleaner.Cleaner {
      private long metadata;

      BatchedMetadataCleaner(long metadata) {
        this.metadata = metadata;
      }

      @Override
      protected boolean cleanImpl(boolean logErrorIfNotClean) {
        boolean neededCleanup = false;
        long address = metadata;
        if (metadata != 0) {
          try {
            NvcompJni.batchedLZ4DecompressDestroyMetadata(metadata);
          } finally {
            // Always mark the resource as freed even if an exception is thrown.
            // We cannot know how far it progressed before the exception, and
            // therefore it is unsafe to retry.
            metadata = 0;
          }
          neededCleanup = true;
        }
        if (neededCleanup && logErrorIfNotClean) {
          log.error("LZ4 BATCHED METADATA WAS LEAKED (Address: " + Long.toHexString(address) + ")");
          logRefCountDebug("Leaked event");
        }
        return neededCleanup;
      }

      @Override
      public boolean isClean() {
        return metadata != 0;
      }
    }
  }
}
