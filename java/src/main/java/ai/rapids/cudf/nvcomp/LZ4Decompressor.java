/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

/** Single-buffer decompression using LZ4 */
public class LZ4Decompressor {

  /**
   * LZ4 decompression settings corresponding to an LZ4 compressed input.
   * NOTE: Each instance must be closed to avoid a native memory leak.
   */
  public static final class Configuration implements AutoCloseable {
    private final long metadataPtr;
    private final long metadataSize;
    private final long tempBytes;
    private final long uncompressedBytes;

    Configuration(long metadataPtr, long metadataSize, long tempBytes,
                  long uncompressedBytes) {
      this.metadataPtr = metadataPtr;
      this.metadataSize = metadataSize;
      this.tempBytes = tempBytes;
      this.uncompressedBytes = uncompressedBytes;
    }

    /** Get the host address of the metadata */
    public long getMetadataPtr() {
      return metadataPtr;
    }

    /** Get the size of the metadata in bytes */
    public long getMetadataSize() {
      return metadataSize;
    }

    /** Get the size of the temporary buffer in bytes needed to decompress */
    public long getTempBytes() {
      return tempBytes;
    }

    /** Get the size of the uncompressed data in bytes */
    public long getUncompressedBytes() {
      return uncompressedBytes;
    }

    @Override
    public void close() {
      NvcompJni.lz4DestroyMetadata(metadataPtr);
    }
  }

  /**
   * Determine if a buffer is data compressed with LZ4.
   * @param buffer data to examine
   * @param stream CUDA stream to use
   * @return true if the data is LZ4 compressed
   */
  public static boolean isLZ4Data(BaseDeviceMemoryBuffer buffer, Cuda.Stream stream) {
    return NvcompJni.isLZ4Data(buffer.getAddress(), buffer.getLength(), stream.getStream());
  }

  /**
   * Get the decompression configuration from compressed data.
   * NOTE: The resulting configuration object must be closed to avoid a native memory leak.
   * @param compressed data that has been compressed by the LZ4 compressor
   * @param stream CUDA stream to use
   * @return decompression configuration for the specified input
   */
  public static Configuration configure(BaseDeviceMemoryBuffer compressed, Cuda.Stream stream) {
    long[] configs = NvcompJni.lz4DecompressConfigure(compressed.getAddress(),
        compressed.getLength(), stream.getStream());
    assert configs.length == 4;
    return new Configuration(configs[0], configs[1], configs[2], configs[3]);
  }

  /**
   * Asynchronously decompress data compressed with the LZ4 compressor.
   * @param compressed buffer containing LZ4-compressed data
   * @param config decompression configuration
   * @param temp temporary storage buffer
   * @param outputBuffer buffer that will be written with the uncompressed output
   * @param stream CUDA stream to use
   */
  public static void decompressAsync(
      BaseDeviceMemoryBuffer compressed,
      Configuration config,
      BaseDeviceMemoryBuffer temp,
      BaseDeviceMemoryBuffer outputBuffer,
      Cuda.Stream stream) {
    NvcompJni.lz4DecompressAsync(
        compressed.getAddress(),
        compressed.getLength(),
        config.getMetadataPtr(),
        config.getMetadataSize(),
        temp.getAddress(),
        temp.getLength(),
        outputBuffer.getAddress(),
        outputBuffer.getLength(),
        stream.getStream());
  }
}
