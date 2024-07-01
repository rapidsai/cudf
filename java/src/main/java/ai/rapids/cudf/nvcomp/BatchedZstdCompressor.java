/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/** Multi-buffer ZSTD compressor */
public class BatchedZstdCompressor extends BatchedCompressor {
  /**
   * Construct a batched ZSTD compressor instance
   * @param chunkSize maximum amount of uncompressed data to compress as a single chunk.
   *                  Inputs larger than this will be compressed in multiple chunks.
   * @param maxIntermediateBufferSize desired maximum size of intermediate device buffers
   *                                  used during compression.
   */
  public BatchedZstdCompressor(long chunkSize, long maxIntermediateBufferSize) {
    super(chunkSize, NvcompJni.batchedZstdCompressGetMaxOutputChunkSize(chunkSize),
        maxIntermediateBufferSize);
  }

  @Override
  protected long batchedCompressGetTempSize(long batchSize, long maxChunkSize) {
    return NvcompJni.batchedZstdCompressGetTempSize(batchSize, maxChunkSize);
  }

  @Override
  protected void batchedCompressAsync(long devInPtrs, long devInSizes, long chunkSize,
      long batchSize, long tempPtr, long tempSize, long devOutPtrs,
      long compressedSizesOutPtr, long stream) {
    NvcompJni.batchedZstdCompressAsync(devInPtrs, devInSizes, chunkSize, batchSize,
        tempPtr, tempSize, devOutPtrs, compressedSizesOutPtr, stream);
  }
}
