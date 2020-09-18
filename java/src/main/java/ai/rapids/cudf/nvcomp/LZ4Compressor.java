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

  public static long getTempSize(BaseDeviceMemoryBuffer input, CompressionType inputType,
                                 long chunkSize) {
    return NvcompJni.lz4CompressGetTempSize(input.getAddress(), input.getLength(),
        inputType.nativeId, chunkSize);
  }

  public static long getOutputSize(BaseDeviceMemoryBuffer input, CompressionType inputType,
                                   long chunkSize, BaseDeviceMemoryBuffer tempBuffer) {
    return NvcompJni.lz4CompressGetOutputSize(input.getAddress(), input.getLength(),
        inputType.nativeId, chunkSize, tempBuffer.getAddress(), tempBuffer.getLength(), false);
  }

  public static long compress(BaseDeviceMemoryBuffer input, CompressionType inputType,
                              long chunkSize, BaseDeviceMemoryBuffer tempBuffer,
                              BaseDeviceMemoryBuffer output, Cuda.Stream stream) {
    return NvcompJni.lz4Compress(input.getAddress(), input.getLength(), inputType.nativeId,
        chunkSize, tempBuffer.getAddress(), tempBuffer.getLength(),
        output.getAddress(), output.getLength(), stream.getStream());
  }

  public static void compressAsync(HostMemoryBuffer compressedSizeOutputBuffer,
                                   BaseDeviceMemoryBuffer input, CompressionType inputType,
                                   long chunkSize, BaseDeviceMemoryBuffer tempBuffer,
                                   BaseDeviceMemoryBuffer output, Cuda.Stream stream) {
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
