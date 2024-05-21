/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

/** LZ4 decompressor that operates on multiple input buffers in a batch */
public class BatchedLZ4Decompressor extends BatchedDecompressor {
  public BatchedLZ4Decompressor(long chunkSize) {
    super(chunkSize);
  }

  /**
   * Asynchronously decompress a batch of buffers
   * @param chunkSize maximum uncompressed block size, must match value used during compression
   * @param origInputs buffers to decompress, will be closed by this operation
   * @param outputs output buffers that will contain the compressed results, each must be sized
   *                to the exact decompressed size of the corresponding input
   * @param stream CUDA stream to use
   *
   * Deprecated: Use the non-static version in the parent class instead.
   */
  public static void decompressAsync(long chunkSize, BaseDeviceMemoryBuffer[] origInputs,
      BaseDeviceMemoryBuffer[] outputs, Cuda.Stream stream) {
    new BatchedLZ4Decompressor(chunkSize).decompressAsync(origInputs, outputs, stream);
  }

  @Override
  protected long batchedDecompressGetTempSize(long numChunks, long maxUncompressedChunkBytes) {
    return NvcompJni.batchedLZ4DecompressGetTempSize(numChunks, maxUncompressedChunkBytes);
  }

  @Override
  protected void batchedDecompressAsync(long devInPtrs, long devInSizes, long devOutSizes,
      long batchSize, long tempPtr, long tempSize, long devOutPtrs, long stream) {
    NvcompJni.batchedLZ4DecompressAsync(devInPtrs, devInSizes, devOutSizes, batchSize, tempPtr,
        tempSize, devOutPtrs, stream);
  }

}
