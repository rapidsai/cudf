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
import ai.rapids.cudf.HostMemoryBuffer;

/** Single-buffer compressor implementing LZ4 */
public class LZ4Compressor {

  /**
   * Calculate the amount of temporary storage space required to compress a buffer.
   * @param input     buffer to compress
   * @param inputType type of data within the buffer
   * @param chunkSize compression chunk size to use
   * @return amount in bytes of temporary storage space required to compress the buffer
   */
  public static long getTempSize(BaseDeviceMemoryBuffer input, CompressionType inputType,
                                 long chunkSize) {
    if (chunkSize <= 0) {
      throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
    }
    return NvcompJni.lz4CompressGetTempSize(input.getAddress(), input.getLength(),
        inputType.nativeId, chunkSize);
  }

  /**
   * Calculate the amount of output storage space required to compress a buffer.
   * @param input      buffer to compress
   * @param inputType  type of data within the buffer
   * @param chunkSize  compression chunk size to use
   * @param tempBuffer temporary storage space
   * @return amount in bytes of output storage space required to compress the buffer
   */
  public static long getOutputSize(BaseDeviceMemoryBuffer input, CompressionType inputType,
                                   long chunkSize, BaseDeviceMemoryBuffer tempBuffer) {
    if (chunkSize <= 0) {
      throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
    }
    return NvcompJni.lz4CompressGetOutputSize(input.getAddress(), input.getLength(),
        inputType.nativeId, chunkSize, tempBuffer.getAddress(), tempBuffer.getLength(), false);
  }

  /**
   * Compress a buffer with LZ4.
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
    return NvcompJni.lz4Compress(input.getAddress(), input.getLength(), inputType.nativeId,
        chunkSize, tempBuffer.getAddress(), tempBuffer.getLength(),
        output.getAddress(), output.getLength(), stream.getStream());
  }

  /**
   * Asynchronously compress a buffer with LZ4. The compressed size output buffer must be pinned
   * memory for this operation to be truly asynchronous. Note that the caller must synchronize
   * on the specified CUDA stream in order to safely examine the compressed output size!
   * @param compressedSizeOutputBuffer host memory where the compressed output size will be stored
   * @param input      buffer to compress
   * @param inputType  type of data within the buffer
   * @param chunkSize  compression chunk size to use
   * @param tempBuffer temporary storage space
   * @param output     buffer that will contain the compressed result
   * @param stream     CUDA stream to use
   */
  public static void compressAsync(HostMemoryBuffer compressedSizeOutputBuffer,
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
    NvcompJni.lz4CompressAsync(compressedSizeOutputBuffer.getAddress(),
        input.getAddress(), input.getLength(), inputType.nativeId, chunkSize,
        tempBuffer.getAddress(), tempBuffer.getLength(), output.getAddress(), output.getLength(),
        stream.getStream());
  }
}
