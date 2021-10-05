/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
import ai.rapids.cudf.HostMemoryBuffer;

/** Single-buffer compressor implementing LZ4 */
public class LZ4Compressor {

  /** LZ4 compression settings corresponding to a chunk size */
  public static final class Configuration {
    private final long metadataBytes;
    private final long tempBytes;
    private final long maxCompressedBytes;

    Configuration(long metadataBytes, long tempBytes, long maxCompressedBytes) {
      this.metadataBytes = metadataBytes;
      this.tempBytes = tempBytes;
      this.maxCompressedBytes = maxCompressedBytes;
    }

    /** Get the size of the metadata information in bytes */
    public long getMetadataBytes() {
      return metadataBytes;
    }

    /** Get the size of the temporary storage in bytes needed to compress */
    public long getTempBytes() {
      return tempBytes;
    }

    /** Get the maximum compressed output size in bytes */
    public long getMaxCompressedBytes() {
      return maxCompressedBytes;
    }
  }

  /**
   * Get the compression configuration necessary for a particular chunk size.
   * @param chunkSize size of an LZ4 chunk in bytes
   * @param uncompressedSize total size of the uncompressed data
   * @return compression configuration for the specified chunk size
   */
  public static Configuration configure(int chunkSize, long uncompressedSize) {
    long[] configs = NvcompJni.lz4CompressConfigure(chunkSize, uncompressedSize);
    assert configs.length == 3;
    return new Configuration(configs[0], configs[1], configs[2]);
  }

  /**
   * Synchronously compress a buffer with LZ4.
   * @param input      buffer to compress
   * @param inputType  type of data within the buffer
   * @param chunkSize  compression chunk size to use
   * @param tempBuffer temporary storage space
   * @param output     buffer that will contain the compressed result
   * @param stream     CUDA stream to use
   * @return size of the resulting compressed data stored to the output buffer
   */
  public static long compress(BaseDeviceMemoryBuffer input, CompressionType inputType,
                              long chunkSize, BaseDeviceMemoryBuffer tempBuffer,
                              BaseDeviceMemoryBuffer output, Cuda.Stream stream) {
    if (chunkSize <= 0) {
      throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
    }
    try (DeviceMemoryBuffer devOutputSizeBuffer = DeviceMemoryBuffer.allocate(Long.BYTES);
         HostMemoryBuffer hostOutputSizeBuffer = HostMemoryBuffer.allocate(Long.BYTES)) {
      compressAsync(devOutputSizeBuffer, input, inputType, chunkSize, tempBuffer, output, stream);
      hostOutputSizeBuffer.copyFromDeviceBuffer(devOutputSizeBuffer, stream);
      return hostOutputSizeBuffer.getLong(0);
    }
  }

  /**
   * Asynchronously compress a buffer with LZ4. The compressed size output buffer must be pinned
   * memory for this operation to be truly asynchronous. Note that the caller must synchronize
   * on the specified CUDA stream in order to safely examine the compressed output size!
   * @param compressedSizeOutputBuffer device memory where the compressed output size will be stored
   * @param input      buffer to compress
   * @param inputType  type of data within the buffer
   * @param chunkSize  compression chunk size to use
   * @param tempBuffer temporary storage space
   * @param output     buffer that will contain the compressed result
   * @param stream     CUDA stream to use
   */
  public static void compressAsync(DeviceMemoryBuffer compressedSizeOutputBuffer,
                                   BaseDeviceMemoryBuffer input, CompressionType inputType,
                                   long chunkSize, BaseDeviceMemoryBuffer tempBuffer,
                                   BaseDeviceMemoryBuffer output, Cuda.Stream stream) {
    if (chunkSize <= 0) {
      throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
    }
    if (compressedSizeOutputBuffer.getLength() < 8) {
      throw new IllegalArgumentException("compressed output size buffer must be able to hold " +
          "at least 8 bytes, size is only " + compressedSizeOutputBuffer.getLength());
    }
    NvcompJni.lz4CompressAsync(
        compressedSizeOutputBuffer.getAddress(),
        input.getAddress(),
        input.getLength(),
        inputType.nativeId,
        chunkSize,
        tempBuffer.getAddress(),
        tempBuffer.getLength(),
        output.getAddress(),
        output.getLength(),
        stream.getStream());
  }
}
